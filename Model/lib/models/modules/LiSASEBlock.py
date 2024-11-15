import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dim="2d"):
        super(SEBlock, self).__init__()
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim =="2d":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Invalid dimension '{dim}' for SEBlock")

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, *([1] * (x.dim() - 2)))
        return x * y.expand_as(x)