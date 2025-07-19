# LiSASEBlock.py
"""
@author   :   andredalwin    
@Contact  :   hello@andredalwin.com
@DateTime :   2025/05/31
@Version  :   1.0
"""
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4, dim="2d"):
        super(SEBlock, self).__init__()
        self.dim = dim  # Store the dimension for use in forward
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == "2d":
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
        y = self.avg_pool(x)
        y = torch.flatten(y, 1)  # Flatten to (batch_size, channels)
        y = self.fc(y)
        if self.dim == "3d":
            y = y.view(y.size(0), y.size(1), 1, 1, 1)  # Reshape for broadcasting
        else:
            y = y.view(y.size(0), y.size(1), 1, 1)  # Reshape for broadcasting
        return x * y


# import torch
# import torch.nn as nn

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=4, dim="2d", threshold=0.5):
#         """
#         Conditional Squeeze-and-Excitation (SE) Block for intelligent feature recalibration
        
#         Args:
#             channels (int): Number of input channels
#             reduction (int): Reduction ratio for channel compression
#             dim (str): Dimensionality of the input ('2d' or '3d')
#             threshold (float): Importance threshold for feature activation
#         """
#         super(SEBlock, self).__init__()
#         self.dim = dim
#         self.threshold = threshold
        
#         # Adaptive pooling based on dimensionality
#         if dim == "3d":
#             self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         elif dim == "2d":
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         else:
#             raise ValueError(f"Invalid dimension '{dim}' for SEBlock")

#         # Feature compression and channel importance estimation
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.GELU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
        
#         # Learnable gating mechanism for adaptive feature recalibration
#         self.gate = nn.Parameter(torch.tensor(1.0))

#     def forward(self, x):
#         """
#         Forward pass with conditional feature recalibration
        
#         Args:
#             x (Tensor): Input feature map
        
#         Returns:
#             Tensor: Recalibrated feature map
#         """
#         # Global feature descriptor
#         y = self.avg_pool(x)
#         y = torch.flatten(y, 1)
#         y = self.fc(y)
        
#         # Compute channel-wise importance
#         importance = y.mean(dim=0)
        
#         # Dynamically decide whether to apply SE block
#         if importance.mean() > self.threshold:
#             # Reshape for broadcasting
#             if self.dim == "3d":
#                 y = y.view(y.size(0), y.size(1), 1, 1, 1)
#             else:
#                 y = y.view(y.size(0), y.size(1), 1, 1)
            
#             # Apply channel-wise scaling with learnable gate
#             return x * (y * self.gate)
        
#         # If importance is low, return original input
#         return x