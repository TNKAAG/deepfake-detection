"""
vit_model.py
------------
ViT-B/16 model for deepfake detection — pretrained on ImageNet.

Fine-tuned in two phases:
  Phase 1: backbone frozen, only the classification head is trained.
  Phase 2: entire network unfrozen, full fine-tuning with a small LR.

Architecture (ViT-B/16):
  Patch size   : 16 x 16
  Patches      : (224 / 16)^2 = 196 patches + 1 class token = 197 tokens
  Hidden dim   : 768
  MLP dim      : 3072
  Heads        : 12
  Layers       : 12
  Total params : ~86 M

Usage:
    from vit_model import get_vit_model, freeze_backbone, unfreeze_backbone
    model = get_vit_model()
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def get_vit_model(num_classes: int = 2) -> nn.Module:
    """
    Returns ViT-B/16 pretrained on ImageNet with the classification
    head replaced by a 2-class linear layer (real / fake).
    """
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    in_features = model.heads.head.in_features          # 768
    model.heads.head = nn.Linear(in_features, num_classes)

    nn.init.xavier_uniform_(model.heads.head.weight)
    nn.init.zeros_(model.heads.head.bias)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classification head (Phase 1)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[Phase 1] Backbone FROZEN — "
        f"trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)"
    )


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning (Phase 2)."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[Phase 2] Backbone UNFROZEN — "
        f"trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)"
    )


if __name__ == "__main__":
    model = get_vit_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (should be (4, 2))")
