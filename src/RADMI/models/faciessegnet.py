"""
FaciesSegNet: U-Net style encoder-decoder for seismic facies segmentation.
"""

import torch
import torch.nn as nn


def double_conv_down(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def double_conv_up(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FaciesSegNet(nn.Module):
    """
    Encoder-decoder network for seismic facies segmentation.
    
    Args:
        n_class: number of output classes
        out_channels: channel sizes for encoder blocks
    """

    def __init__(self, n_class: int, out_channels: tuple = (8, 10, 30, 40, 60)):
        super().__init__()
        
        self.n_class = n_class
        self.out_channels = out_channels

        # encoder
        in_ch = 1
        self.down_convs = nn.ModuleList()
        for out_ch in out_channels:
            self.down_convs.append(double_conv_down(in_ch, out_ch))
            in_ch = out_ch

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # decoder
        self.up_convs = nn.ModuleList()
        reversed_channels = list(reversed(out_channels))
        for i in range(len(reversed_channels) - 1):
            self.up_convs.append(
                double_conv_up(reversed_channels[i], reversed_channels[i + 1])
            )

        # output heads
        self.conv_last = nn.Conv2d(out_channels[0], n_class, kernel_size=1)
        self.conv_reconstruct = nn.Conv2d(out_channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2], x.shape[3]

        # encoder
        for i, block in enumerate(self.down_convs):
            x = block(x)
            if i < len(self.down_convs) - 1:
                x = self.maxpool(x)

        # decoder
        for block in self.up_convs:
            x = block(x)

        # crop to input size
        out = self.conv_last(x)[:, :, :h, :w]
        reconstruct = self.conv_reconstruct(x)[:, :, :h, :w]

        return out, reconstruct
