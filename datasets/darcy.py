import os
import random

import h5py
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('darcy')
class DarcyDataset(Dataset):
    def __init__(self, root, split, data_file, bc_range=(-2., 2.), n_data=10000, scale=100.):
        self.root = root
        self.split = split
        self.data_file = data_file
        self.file = h5py.File(os.path.join(root, data_file), 'r')
        self.bc_range = bc_range
        self.n_data = n_data
        self.scale = scale

        self.data = self.file['sol']  # (n, s, s)
        self.n_dataset, self.s, _ = self.data.shape
        assert n_data % self.n_dataset == 0, \
            f'`n_data` ({n_data}) must be divisible by the true dataset size ({self.n_dataset})'

    def __del__(self):
        self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        index = index % self.n_dataset
        data = (torch.from_numpy(self.data[index]) * self.scale).to(torch.float32)
        bc = random.uniform(*self.bc_range)
        return data + bc
