import random

import torch
from torch.utils.data import Dataset
import numpy as np

from ._base import register_dataset


@register_dataset('stokes')
class StokesDataset(Dataset):
    r"""
    1D Stokes second problem.
    .. math::
        u_t = \nu u_{yy}, 0 \le y \le 1, 0 \le t \le 1
    .. math::
        u_0(y) = U \exp(-k y) \cos(k y)
    .. math::
        u(y=0, t) = U \cos(\omega t)

    The exact solution is :math:`u(y, t) = U \exp(-k y) \cos(k y - \omega t)`
    """

    def __init__(self, split, nx=100, nt=100, visc_range=(0.01, 1.), k_range=None,
                 freq_range=(2., 8.), t_range=(0., 1.), amp=2., n_data=5000):
        self.split = split
        self.nx = nx
        self.nt = nt
        self.visc_range = visc_range
        self.k_range = k_range
        self.freq_range = freq_range
        self.t_range = t_range
        self.amp = amp
        self.n_data = n_data

        self.xs = torch.linspace(0., 1., nx).view(-1, 1)
        self.ts = torch.linspace(*t_range, nt).view(1, -1)

    def __getitem__(self, index):
        freq = random.uniform(*self.freq_range)
        if self.k_range is None:
            v = random.uniform(*self.visc_range)
            k = np.sqrt(freq / (2 * v))
        else:
            k = random.uniform(*self.k_range)
        y = self.amp * torch.exp(-k * self.xs) * torch.cos(k * self.xs - freq * self.ts)  # (nx, nt)
        return y

    def __len__(self):
        return self.n_data
