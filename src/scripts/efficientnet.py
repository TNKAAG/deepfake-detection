"""
efficientnet.py
---------------
EfficientNet-B0 fine-tuned for binary deepfake detection.

Two-phase training strategy:
  Phase 1 — Backbone FROZEN.  Only the classifier head trains.
             Lets the head stabilise before touching pretrained weights.
  Phase 2 — Backbone UNFROZEN. Everything trains at a much lower LR.
             Adapts ImageNet features to deepfake-specific patterns.

Usage:
    from efficientnet import get_efficientnet, freeze_backbone, unfreeze_backbone
    model = get_efficientnet()
    freeze_backbone(model)    # Phase 1
    unfreeze_backbone(model)  # Phase 2
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeEfficientNet(nn.Module):
    """
    EfficientNet-B0 with a custom 2-layer classifier head.

    The original ImageNet head (1000 classes) is replaced with:
        Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2)

    1280 is EfficientNet-B0's native feature dimension — do not change this.

    Input:  (B, 3, 224, 224)
    Output: (B, 2)  — logits for [fake, real]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        # Load with latest ImageNet weights — much better than 'pretrained=True'
        # which is deprecated. IMAGENET1K_V1 is the standard benchmark weights.
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # ── Backbone (all layers except the classifier) ───────────
        # EfficientNet splits cleanly into .features and .classifier
        self.backbone = base.features        # (B, 3, 224, 224) → (B, 1280, 7, 7)
        self.pool     = base.avgpool         # (B, 1280, 7, 7)  → (B, 1280, 1, 1)

        # ── Custom classifier head ────────────────────────────────
        # Two layers gives enough capacity without overfitting.
        # Dropout is slightly lower than baseline (0.4 vs 0.5) because
        # the pretrained backbone already provides strong regularisation.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Initialise only the new head — backbone keeps ImageNet weights
        self._init_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)              # (B, 1280, 7, 7)
        x = self.pool(x)                  # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)           # (B, 1280)
        x = self.classifier(x)            # (B, 2)
        return x

    def _init_head(self):
        """Kaiming init on the new head layers only."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────
# PHASE CONTROL HELPERS
# ─────────────────────────────────────────────

def freeze_backbone(model: DeepfakeEfficientNet):
    """
    PHASE 1 — Freeze all backbone parameters.
"""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.pool.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Phase 1] Backbone FROZEN — "
          f"trainable: {trainable:,} / {total:,} parameters "
          f"({100*trainable/total:.1f}%)")


def unfreeze_backbone(model: DeepfakeEfficientNet):
    """
    PHASE 2 — Unfreeze all backbone parameters.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.pool.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Phase 2] Backbone UNFROZEN — "
          f"trainable: {trainable:,} / {total:,} parameters "
          f"({100*trainable/total:.1f}%)")


def get_efficientnet(num_classes: int = 2, dropout: float = 0.4) -> DeepfakeEfficientNet:
    """ single clean import for the notebook."""
    return DeepfakeEfficientNet(num_classes=num_classes, dropout=dropout)


def count_params(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}



if __name__ == "__main__":
    model = get_efficientnet()

    print("── Phase 1 (head only) ──────────────────────────────────")
    freeze_backbone(model)
    p = count_params(model)
    print(f"   Total: {p['total']:,}   Trainable: {p['trainable']:,}")

    print("\n── Phase 2 (full fine-tune) ─────────────────────────────")
    unfreeze_backbone(model)
    p = count_params(model)
    print(f"   Total: {p['total']:,}   Trainable: {p['trainable']:,}")

    # Verify forward pass shape
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"\nInput:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (should be (4, 2))")
