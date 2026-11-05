"""A small U-Net: photo in -> lane-pixel map out."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 'looking' layers in a row: conv -> normalize -> ReLU, twice."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        c0, c1, c2, c3, c4 = base, base * 2, base * 4, base * 8, base * 16

        # Going DOWN (zoom out)
        self.enc1 = DoubleConv(in_channels, c0)
        self.enc2 = DoubleConv(c0, c1)
        self.enc3 = DoubleConv(c1, c2)
        self.enc4 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        # Going UP (zoom in)
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec4 = DoubleConv(c3 * 2, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = DoubleConv(c2 * 2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2 = DoubleConv(c1 * 2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.dec1 = DoubleConv(c0 * 2, c0)

        self.head = nn.Conv2d(c0, 1, kernel_size=1)   # 1 number per pixel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # torch.cat = the "sticky note" skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)          # raw scores ("logits"), not yet 0..1
