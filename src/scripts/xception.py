"""
xception.py
-----------
Xception model for binary deepfake detection.
Custom implementation based on the original Xception architecture,
trained from scratch by Shaun in Google Colab.

Architecture:
  Entry flow    → 3 blocks with depthwise separable convs, halving spatial dims
  Middle flow   → 8 residual blocks maintaining 728 channels
  Exit flow     → upscale to 2048 channels, GAP, classifier

Usage:
    from xception import get_xception
    model = get_xception()
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise → pointwise → BN → optional ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False, activate_first=True):
        super().__init__()
        self.activate_first = activate_first
        self.relu      = nn.ReLU()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn        = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.activate_first:
            x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class EntryFlowBlock(nn.Module):
    """Two SeparableConvs + MaxPool + 1x1 skip. Halves spatial dims."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep_convs = nn.Sequential(
            SeparableConv2d(in_channels,  out_channels, activate_first=True),
            SeparableConv2d(out_channels, out_channels, activate_first=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.sep_convs(x) + self.skip(x)


class MiddleFlowBlock(nn.Module):
    """Three SeparableConvs + residual, no downsampling. Repeated 8x."""

    def __init__(self, channels=728):
        super().__init__()
        self.sep_convs = nn.Sequential(
            SeparableConv2d(channels, channels, activate_first=True),
            SeparableConv2d(channels, channels, activate_first=True),
            SeparableConv2d(channels, channels, activate_first=True),
        )

    def forward(self, x):
        return self.sep_convs(x) + x


class ExitFlowBlock(nn.Module):
    """728→1024 with MaxPool + 1x1 skip. Halves spatial dims."""

    def __init__(self):
        super().__init__()
        self.sep_convs = nn.Sequential(
            SeparableConv2d(728,  728,  activate_first=True),
            SeparableConv2d(728,  1024, activate_first=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )

    def forward(self, x):
        return self.sep_convs(x) + self.skip(x)


class Xception(nn.Module):
    """Xception for binary deepfake detection. Input: (B,3,224,224) → Output: (B,2)"""

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.entry_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.entry_block1 = EntryFlowBlock(64,  128)
        self.entry_block2 = EntryFlowBlock(128, 256)
        self.entry_block3 = EntryFlowBlock(256, 728)
        self.middle_flow  = nn.Sequential(*[MiddleFlowBlock(728) for _ in range(8)])
        self.exit_block   = ExitFlowBlock()
        self.exit_sep1    = SeparableConv2d(1024, 1536, activate_first=False)
        self.exit_sep2    = SeparableConv2d(1536, 2048, activate_first=False)
        self.exit_relu    = nn.ReLU()
        self.gap          = nn.AdaptiveAvgPool2d(1)
        self.classifier   = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2048, num_classes))
        self._init_weights()

    def forward(self, x):
        x = self.entry_stem(x)
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)
        x = self.middle_flow(x)
        x = self.exit_block(x)
        x = self.exit_sep1(x)
        x = self.exit_sep2(x)
        x = self.exit_relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def get_xception(num_classes: int = 2, dropout: float = 0.5) -> Xception:
    """Factory function — single clean import for the notebook."""
    return Xception(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model = get_xception()
    total = sum(p.numel() for p in model.parameters())
    print(f"Xception — {total:,} parameters")

    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (should be (2, 2))")
