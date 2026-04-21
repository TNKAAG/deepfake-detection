"""
frequency_cnn.py
----------------
Frequency-domain CNN for deepfake detection.
Custom implementation by George that operates on DCT-transformed grayscale images
rather than raw RGB pixels.

Key difference from other models:
  Input: (B, 1, 128, 128) — 2D DCT coefficients, log-scaled
  Other models: (B, 3, 224, 224) — RGB pixels

This targets spectral artifacts invisible in pixel space — GAN-based
deepfake methods leave characteristic patterns in the frequency domain.

Usage:
    from frequency_cnn import get_frequency_cnn, FrequencyTransform
    
    # Use FrequencyTransform instead of the standard transforms
    transform = FrequencyTransform()
    model = get_frequency_cnn()
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import dct
from PIL import Image


class FrequencyTransform:
    """
    Transform pipeline for frequency-domain CNN.
    Converts RGB PIL Image → grayscale → 2D DCT → log-scaled tensor.
    
    This replaces the standard torchvision transforms for George's model.
    """
    def __init__(self, size=128):
        self.size = size
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Args:
            img: PIL Image (RGB)
        Returns:
            torch.Tensor of shape (1, 128, 128) — single-channel DCT coefficients
        """
        # PIL Image → numpy (already RGB)
        img_np = np.array(img)
        
        # RGB → grayscale
        if len(img_np.shape) == 3:
            # Using standard ITU-R 601-2 luma transform
            gray = (0.299 * img_np[:,:,0] + 
                   0.587 * img_np[:,:,1] + 
                   0.114 * img_np[:,:,2]).astype(np.uint8)
        else:
            gray = img_np
        
        # Resize to 128×128
        from PIL import Image as PILImage
        gray_img = PILImage.fromarray(gray)
        gray_img = gray_img.resize((self.size, self.size), PILImage.LANCZOS)
        gray = np.array(gray_img)
        
        # 2D DCT (discrete cosine transform)
        dct_transformed = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        
        # Log scaling to compress dynamic range
        epsilon = 1e-12
        dct_log = np.log(np.abs(dct_transformed) + epsilon)
        
        # Convert to tensor: (128, 128) → (1, 128, 128)
        tensor = torch.from_numpy(dct_log.astype(np.float32)).unsqueeze(0)
        
        return tensor


class FreqCNN(nn.Module):
    """
    Frequency-domain CNN.
    
    Input:  (B, 1, 128, 128) — DCT coefficients
    Output: (B, 2) — logits for [fake, real]
    
    Architecture:
      3 conv blocks (1→32→64→128) with BN + ReLU + MaxPool
      Flatten → FC(256) → Dropout(0.5) → FC(2)
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 128 → 64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 64 → 32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 32 → 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        self._init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


def get_frequency_cnn(num_classes: int = 2, dropout: float = 0.5) -> FreqCNN:
    """Factory function — single clean import for the notebook."""
    return FreqCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model = get_frequency_cnn()
    total = sum(p.numel() for p in model.parameters())
    print(f"FrequencyCNN — {total:,} parameters")
    
    # Test with DCT input shape
    dummy = torch.randn(2, 1, 128, 128)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (should be (2, 2))")
