"""U-Net architectures for 2D and 3D segmentation."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net for slice-wise segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 32) -> None:
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock2D(in_channels, f)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.enc4 = ConvBlock2D(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock2D(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(f * 2, f)
        self.out = nn.Conv2d(f, num_classes, kernel_size=1)

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
        return self.out(d1)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 16) -> None:
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.enc4 = ConvBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(f * 8, f * 16)
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)
        self.out = nn.Conv3d(f, num_classes, kernel_size=1)

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
        return self.out(d1)


def build_model(arch: str, in_channels: int = 1, num_classes: int = 1, base_filters: int | None = None) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "unet2d":
        return UNet2D(in_channels, num_classes, base_filters or 32)
    if arch == "unet3d":
        return UNet3D(in_channels, num_classes, base_filters or 16)
    raise ValueError(f"Unknown architecture: {arch}")


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
