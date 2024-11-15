# LiSASEBlock.py
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dim="2d"):
        super(SEBlock, self).__init__()
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
            self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        elif dim == "2d":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
            self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        else:
            raise ValueError(f"Invalid dimension '{dim}' for SEBlock")

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y