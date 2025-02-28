import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.special import erf
from scipy.optimize import root_scalar

from ._base import register_dataset


@register_dataset('stefan')
class StefanDataset(Dataset):
    def __init__(self, split, nx=100, nt=100, k_range=(0.5, 2.), shock_range=(0.3, 0.9),
                 t_range=(0., 1.), n_data=5000):
        self.split = split
        self.nx = nx
        self.nt = nt
        self.k_range = k_range
        self.shock_range = shock_range
        self.t_range = t_range
        self.n_data = n_data

        self.xs = torch.linspace(0., 1., nx).view(-1, 1)
        self.ts = torch.linspace(*t_range, nt).view(1, -1)

    def z1(self, shock):
        def f(_z1):
            a1 = shock * erf(_z1)
            a2 = _z1 * np.exp(_z1 ** 2)
            b = (1 - shock) / np.sqrt(np.pi)
            return a1 * a2 - b

        return root_scalar(f, bracket=[0, 10]).root

    def c1(self, shock, kmax):
        alpha = 2 * np.sqrt(kmax) * self.z1(shock)
        return (1 - shock) / erf(alpha / (2 * np.sqrt(kmax)))

    def __getitem__(self, index):
        kmax = random.uniform(*self.k_range)
        shock = random.uniform(*self.shock_range)

        alpha = 2 * np.sqrt(kmax) * self.z1(shock)
        c1 = (1 - shock) / erf(alpha / (2 * np.sqrt(kmax)))
        a = self.xs / (2 * torch.sqrt(self.ts * kmax))
        p1 = 1 - c1 * torch.erf(a)
        x_star = alpha * torch.sqrt(self.ts)
        p = torch.where(self.xs <= x_star, p1, torch.zeros_like(self.xs))
        p[torch.isclose(self.xs, torch.zeros_like(self.xs)).expand_as(p)] = 1.0
        return p.squeeze(-1)

    def __len__(self):
        return self.n_data
