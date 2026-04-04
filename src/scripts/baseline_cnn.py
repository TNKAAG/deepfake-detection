"""
baseline_cnn.py
---------------
Baseline CNN model for deepfake detection — trained from scratch.

This is the BASELINE model. Its job is to establish a performance
floor that the EfficientNet-B0 transfer learning model will be
compared against in the final report.

Architecture:
  4 x ConvBlock (Conv → BN → ReLU → MaxPool)
  Global Average Pooling
  2-layer classifier head

Usage:
    from baseline_cnn import get_baseline_model
    model = get_baseline_model()
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU → MaxPool2d

    BatchNorm is critical here — without pretrained weights,
    raw CNNs are very sensitive to internal covariate shift.
    bias=False on Conv because BatchNorm subsumes it.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Baseline CNN for binary deepfake detection (real vs fake).

    Input:  (B, 3, 224, 224)
    Output: (B, 2)  — logits for [fake, real]

    Spatial resolution after each block:
      Input:   224 x 224
      Block 1: 112 x 112  (32 maps)
      Block 2:  56 x 56   (64 maps)
      Block 3:  28 x 28   (128 maps)
      Block 4:  14 x 14   (256 maps)
      GAP:       1 x 1    (256-dim vector)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
        )

        # Global Average Pooling — preferred over Flatten+Linear because:
        #   1. Far fewer parameters → less overfitting without pretraining
        #   2. Forces spatially meaningful feature maps
        #   3. Produces cleaner GradCAM visualisations in evaluation
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Kaiming (He) initialisation — correct choice for ReLU networks.
        # Without this, training from scratch converges very slowly.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


def get_baseline_model(num_classes: int = 2, dropout: float = 0.5) -> BaselineCNN:
    """Factory function — single clean import for the notebook."""
    return BaselineCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model        = get_baseline_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    dummy = torch.randn(8, 3, 224, 224)
    out   = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (should be (8, 2))")
