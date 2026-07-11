"""U-Net encoder-decoder for landmark heatmap regression (Stage 2B).

Input: 640x640x1 grayscale OPG.
Output: 640x640x3 heatmap stack (channel 0=bone crest, 1=sinus floor, 2=nerve canal),
sigmoid-activated so each channel is bounded to [0, 1] matching Gaussian heatmap targets.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_LANDMARK_CHANNELS = 3


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetLandmark(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = NUM_LANDMARK_CHANNELS, base_filters: int = 32):
        super().__init__()
        f = base_filters

        self.enc1 = _ConvBlock(in_channels, f)
        self.enc2 = _ConvBlock(f, f * 2)
        self.enc3 = _ConvBlock(f * 2, f * 4)
        self.enc4 = _ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(f * 2, f)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))
