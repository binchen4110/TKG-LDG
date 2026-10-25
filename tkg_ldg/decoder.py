import torch
from torch import nn
from torch.nn import functional as F


class ConvTransDecoder(nn.Module):
    """Compose two query embeddings by convolution and score all candidates (Eqs. 7–8)."""

    def __init__(self, dim, channels=50, kernel_size=3, dropout=0.2):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("Decoder kernel size must be a positive odd number")
        self.bn0 = nn.BatchNorm1d(2)
        self.conv1 = nn.Conv1d(2, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc = nn.Linear(dim * channels, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.inp_drop = nn.Dropout(dropout)
        self.feature_map_drop = nn.Dropout(dropout)
        self.hidden_drop = nn.Dropout(dropout)

    def forward(self, first, second, candidates):
        x = self.inp_drop(self.bn0(torch.stack((first, second), dim=1)))
        x = self.feature_map_drop(F.relu(self.bn1(self.conv1(x))))
        x = self.hidden_drop(self.fc(x.flatten(1)))
        if len(x) > 1 or not self.training:
            x = self.bn2(x)
        return F.relu(x) @ candidates.T
