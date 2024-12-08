"""Architecture of attention model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Attention model class."""

    def __init__(self, num_class: int) -> None:
        """Intialize the model layers."""
        super(AttentionHead, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_features=512)
        self.conv1 = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=1)

        # AttentionMap2D
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )

        # UpscaleAttention
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=num_class)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass for the attention head."""
        print(x.shape)
        x = self.bn1(x)
        bn_residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        upscale_attention_residual = x

        # multiply the upscale_attention_residual and bn_residual
        x = torch.mul(x, bn_residual)
        x = torch.mean(x, dim=(2, 3))

        # now use upscale residual
        y = torch.mean(upscale_attention_residual, dim=(2, 3))

        # Add both output
        x += y

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x