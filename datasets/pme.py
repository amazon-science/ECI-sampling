import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('pme')
class PMEDataset(Dataset):
    def __init__(self, split, nx=100, nt=100, m_range=(1., 6.), t_range=(0., 1.), n_data=5000):
        self.split = split
        self.nx = nx
        self.nt = nt
        self.m_range = m_range
        self.t_range = t_range
        self.n_data = n_data

        self.xs = torch.linspace(0., 1., nx).view(-1, 1)
        self.ts = torch.linspace(*t_range, nt).view(1, -1)
        self.indicator = F.relu(self.ts - self.xs)

    def __getitem__(self, index):
        m = random.uniform(*self.m_range)
        return (m * self.indicator).pow(1 / m)

    def __len__(self):
        return self.n_data
