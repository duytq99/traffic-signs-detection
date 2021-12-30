import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def visualize(self):
        pass

    def info(self):
        pass