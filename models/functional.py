from __future__ import annotations

import torch
import torch.nn as nn
import gpytorch
from torchdiffeq import odeint
from tqdm import tqdm

from .constraints import Constraint, NoneConstraint


class GPPrior(gpytorch.models.ExactGP):
    """ Wrapper around some gpytorch utilities that makes prior sampling easy.
    """

    def __init__(self, kernel: str | None = None, mean=None, lengthscale=None, var=None):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        """

        # Initialize parent module; requires a likelihood so small hack
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(None, None, likelihood)
        self.kernel = kernel

        if mean is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean

        if kernel is None or kernel.lower() == 'matern':
            eps = 1e-10  # Diagonal covariance jitter
            nu = 0.5  # Smoothness parameter, in [0.5, 1.5, 2.5]

            # Default settings for length/variance
            if lengthscale is None:
                self.lengthscale = torch.tensor([0.01])
            else:
                self.lengthscale = torch.as_tensor(lengthscale)
            if var is None:
                self.outputscale = torch.tensor([0.1])  # Variance
            else:
                self.outputscale = torch.as_tensor(var)

            # Create Matern kernel with appropriate lengthscale
            base_kernel = gpytorch.kernels.MaternKernel(nu, eps=eps)
            base_kernel.lengthscale = self.lengthscale

            # Wrap with ScaleKernel to get appropriate variance
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.covar_module.outputscale = self.outputscale

        elif kernel.lower() in ('randn', 'rand'):
            self.covar_module = None
        else:
            raise ValueError(f'Unknown kernel: {kernel}')

        self.eval()  # Required for sampling from prior

    def check_input(self, x, dims=None):
        assert x.ndim == 2, f'Input {x.shape} should have shape (n_points, dim)'
        if dims:
            assert x.shape[1] == len(dims), f'Input {x.shape} should have shape (n_points, dim)'

    def forward(self, x):
        """ Creates a Normal distribution at the points in x.
        x: locations to query at, a flattened grid; tensor (n_points, dim)

        returns: a gpytorch distribution corresponding to a Gaussian at x
        """
        self.check_input(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def sample(self, x, dims, n_samples=1):
        """ Draws samples from the GP prior.
        x: locations to sample at, a flattened grid; tensor (n_points, n_dim)
        dims: list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
        n_samples: number of samples to draw

        returns: samples from the GP; tensor (n_samples, dims[0], dims[1], ...)
        """
        self.eval()
        self.check_input(x, dims)
        if self.kernel.lower() == 'randn':
            return torch.randn(n_samples, *dims, device=x.device)
        if self.kernel.lower() == 'rand':
            return torch.rand(n_samples, *dims, device=x.device)

        distr = self(x)
        samples = distr.sample(sample_shape=torch.Size([n_samples, ]))
        samples = samples.reshape(n_samples, *dims)
        return samples


def make_grid(dims: tuple[int], device='cpu', start: float | tuple[float] = 0., end: float | tuple[float] = 1.):
    ndim = len(dims)
    if not isinstance(start, (tuple, list)):
        start = [start] * ndim
    if not isinstance(end, (tuple, list)):
        end = [end] * ndim
    if ndim == 1:
        return torch.linspace(start[0], end[0], dims[0], dtype=torch.float, device=device).unsqueeze(-1)
    xs = torch.meshgrid([
        torch.linspace(start[i], end[i], dims[i], dtype=torch.float, device=device)
        for i in range(ndim)
    ], indexing='ij')
    grid = torch.stack(xs, dim=-1).view(-1, ndim)
    return grid


class FFM(nn.Module):
    """
    Functional Flow Matching (FFM) model for generative modeling of PDE solutions.
    """

    def __init__(self, model, kernel='matern', kernel_length=0.001, kernel_variance=1.0):
        super().__init__()
        self.model = model
        self.gp = GPPrior(kernel=kernel, lengthscale=kernel_length, var=kernel_variance)

    def forward(self, t, xt):
        pred = self.model(t, xt)
        return pred

    def simulate(self, t, x):
        batch_size, *dims = x.size()
        grid = make_grid(dims, x.device)
        noise = self.gp.sample(grid, dims, n_samples=batch_size)
        t = t[(slice(None),) + (None,) * (x.dim() - 1)]
        return (1 - t) * noise + t * x, x - noise

    def get_loss(self, x):
        t = torch.rand(x.size(0), dtype=torch.float, device=x.device)
        xt, vf = self.simulate(t, x)
        pred_vf = self(t, xt)
        loss = (pred_vf - vf).pow(2).mean()
        return loss

    @torch.no_grad()
    def sample(self, n_sample, n_eval, dims, device, rtol=1e-5, atol=1e-5, return_traj=False):
        """
        Sample from the FFM model by integrating the GP prior with the learned vector field.
        :param n_sample: number of samples to generate
        :param n_eval: number of evaluation steps
        :param dims: dimensions of the generated grid
        :param device: device to run on
        :param rtol: relative tolerance for ODE solver
        :param atol: absolute tolerance for ODE solver
        :param return_traj: whether to return the full trajectory
        :return: generated samples of shape (n_sample, *dims) or trajectory of shape (n_eval + 1, n_sample, *dims)
        """
        grid = make_grid(dims, device)
        noise = self.gp.sample(grid, dims, n_samples=n_sample)
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
        xs = odeint(self.model, noise, ts, method='dopri5', rtol=rtol, atol=atol)
        if return_traj:
            return xs.detach()
        return xs[-1].detach()

    @torch.no_grad()
    def eci_sample(
            self, n_sample: int, n_step: int, n_mix: int, resample_step: int | None,
            dims: tuple[int], device: str, constraint: Constraint = NoneConstraint()
    ):
        """
        Guided sampling by mixing the information between the constrained and unconstrained regions with
        interleaved interpolation, correction, and extrapolation.
        :param n_sample: number of samples to generate
        :param n_step: number of Euler steps
        :param n_mix: number of mixing steps in each Euler step
        :param resample_step: resample interval for Gaussian noise, if 0 or None, no noise resampling
        :param dims: dimensions of the generated grid
        :param device: device to run on
        :param constraint: constraint to apply, default is NoneConstraint
        :return: generated samples of shape (n_sample, *dims)
        """
        grid = make_grid(dims, device)
        noise = self.gp.sample(grid, dims, n_samples=n_sample)
        x = noise
        ts = torch.linspace(0, 1, n_step + 1, device=device)
        cnt = 0
        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1
        dt = 1 / n_step
        for t in tqdm(ts[:-1], desc='ECI Sampling'):
            for u in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    noise = self.gp.sample(grid, dims, n_samples=n_sample)
                vf = self.model(t, x)
                x1 = x + vf * (1 - t)
                x1 = constraint.adjust(x1)
                if u < n_mix - 1:
                    x = x1 * t + noise * (1 - t)
                else:
                    x = x1 * (t + dt) + noise * (1 - t - dt)
        return x.detach()

    def guided_sample(self, value, mask, n_sample, n_step, eta=2e2):
        """
        Gradient guided sampling from DiffusionPDE. https://arxiv.org/abs/2406.17763
        :param value: constraint values
        :param mask: constraint mask
        :param n_sample: number of samples to generate
        :param n_step: number of Euler steps
        :param eta: gradient coefficient
        :return: generated samples of shape (n_sample, *dims)
        """
        grid = make_grid(value.size(), value.device)
        noise = self.gp.sample(grid, value.size(), n_samples=n_sample)
        x = noise
        ts = torch.linspace(0, 1, n_step + 1, device=value.device)
        value, mask = value[None], mask[None]
        for t in tqdm(ts[:-1], desc='Guided sampling'):
            vf = self.model(t, x).detach()
            if t < ts[-2]:
                # 2nd order correlation
                vf2 = self.model(t + 1 / n_step, x).detach()
                vf = (vf + vf2) / 2

            # observation loss
            x.requires_grad_(True)
            x1 = x + vf * (1 - t)
            loss_obs = ((x1 - value) * mask).square().sum() / mask.sum() / n_sample
            loss_obs.backward()
            grad = x.grad
            x = x.detach() + vf / n_step - eta * grad
        return x.detach()

    def dflow_sample(self, value, mask, n_sample, n_step, n_iter=20, lr=1e-1):
        """
        D-Flow sampling by differentiating through the ODE solver. https://arxiv.org/abs/2402.14017
        :param value: constraint values
        :param mask: constraint mask
        :param n_sample: number of samples to generate
        :param n_step: number of Euler steps
        :param n_iter: number of optimization iterations
        :param lr: learning rate
        :return: generated samples of shape (n_sample, *dims)
        """
        cnt = 0

        def loss_fn(x0):
            x = x0
            ts = torch.linspace(0, 1, n_step + 1, device=value.device)
            for t in ts[:-1]:
                vf = self.model(t, x)
                x = x + vf / n_step
            loss = ((x - value) * mask).square().sum()
            return x, loss

        def closure():
            nonlocal cnt
            cnt += 1
            optimizer.zero_grad()
            _, loss = loss_fn(noise)
            loss.backward()
            print(f'Iter {cnt}: {loss.item():.4f}')
            return loss

        grid = make_grid(value.size(), value.device)
        noise = self.gp.sample(grid, value.size(), n_samples=n_sample)
        noise.requires_grad_(True)
        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        x1, _ = loss_fn(noise.detach())
        return x1


class ConditionalFFM(nn.Module):
    """
    Conditional Functional Flow Matching model for generative modeling of PDE solutions
    with additional conditional values as input in both training and inference.
    """

    def __init__(self, model, kernel='matern', kernel_length=0.001, kernel_variance=1.0):
        super().__init__()
        self.model = model
        self.gp = GPPrior(kernel=kernel, lengthscale=kernel_length, var=kernel_variance)

    def forward(self, t, xt, cond):
        pred = self.model(t, xt, cond)
        return pred

    def simulate(self, t, x):
        batch_size, *dims = x.size()
        grid = make_grid(dims, x.device)
        noise = self.gp.sample(grid, dims, n_samples=batch_size)
        t = t[(slice(None),) + (None,) * (x.dim() - 1)]
        return (1 - t) * noise + t * x, x - noise

    def get_loss(self, x, cond):
        t = torch.rand(x.size(0), dtype=torch.float, device=x.device)
        xt, vf = self.simulate(t, x)
        pred_vf = self(t, xt, cond)
        loss = (pred_vf - vf).pow(2).mean()
        return loss

    @torch.no_grad()
    def sample(self, cond, n_eval, rtol=1e-5, atol=1e-5, return_traj=False):
        device = cond.device
        n_sample, *dims = cond.size()
        grid = make_grid(dims, device)
        noise = self.gp.sample(grid, dims, n_samples=n_sample)
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
        xs = odeint((lambda t, x: self.model(t, x, cond)), noise, ts, method='dopri5', rtol=rtol, atol=atol)
        if return_traj:
            return xs.detach()
        return xs[-1].detach()
