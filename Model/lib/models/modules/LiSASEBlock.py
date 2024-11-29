# # LiSASEBlock.py
# import torch
# import torch.nn as nn

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=4, dim="2d"):
#         super(SEBlock, self).__init__()
#         self.dim = dim  # Store the dimension for use in forward
#         if dim == "3d":
#             self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         elif dim == "2d":
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         else:
#             raise ValueError(f"Invalid dimension '{dim}' for SEBlock")

#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = torch.flatten(y, 1)  # Flatten to (batch_size, channels)
#         y = self.fc(y)
#         if self.dim == "3d":
#             y = y.view(y.size(0), y.size(1), 1, 1, 1)  # Reshape for broadcasting
#         else:
#             y = y.view(y.size(0), y.size(1), 1, 1)  # Reshape for broadcasting
#         return x * y

# LiSASEBlock.py
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8, dim="2d"):
        super(SEBlock, self).__init__()
        self.dim = dim  # Store the dimension for use in forward
        
        # Adaptive pooling for 2D or 3D input
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == "2d":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Invalid dimension '{dim}' for SEBlock")

        # Excitation: Add a larger reduction factor and use GELU instead of ReLU
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),  # Use GELU for smoother activation
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # Residual connection for stability
        self.residual = nn.Identity()  # Identity map for skip connection

    def forward(self, x):
        # Squeeze: Global average pooling
        y = self.avg_pool(x)  # Apply average pooling

        # Flatten the pooled output
        y = torch.flatten(y, 1)  # Flatten to (batch_size, channels)

        # Excitation: Apply the fully connected layers and the activation
        y = self.fc(y)

        # Reshape for broadcasting (either 2D or 3D)
        if self.dim == "3d":
            y = y.view(y.size(0), y.size(1), 1, 1, 1)  # Reshape for 3D input
        else:
            y = y.view(y.size(0), y.size(1), 1, 1)  # Reshape for 2D input

        # Apply the channel attention weights and add residual connection
        out = x * y + self.residual(x)  # Apply scaling and residual
        return out
