"""
src/model_unet.py
=================
Baseline U-Net for MRI → CT image synthesis.

Architecture overview
---------------------
The U-Net is the standard encoder–decoder network with skip connections
originally proposed by Ronneberger et al. (2015) for biomedical image
segmentation, and widely adopted for image-to-image translation tasks.

Key ideas:
  - Encoder (4 blocks): progressively down-samples the spatial resolution
    while increasing the number of feature channels.  This forces the network
    to learn a compact, abstract representation of the input.
  - Bottleneck: deepest representation, no spatial change.
  - Decoder (4 blocks): progressively up-samples back to the original
    resolution.  Skip connections concatenate encoder features from the
    matching resolution, letting the decoder recover fine spatial details
    that would otherwise be lost in the bottleneck.
  - Output: single-channel tanh activation → values in [−1, 1] to match
    the normalized CT range.

Input/Output
------------
  Input:  (B, 1, 256, 256) — single-channel normalized MRI slice
  Output: (B, 1, 256, 256) — single-channel synthetic CT slice in [−1, 1]

Reference
---------
  Ronneberger, O., Fischer, P., & Brox, T. (2015).
  U-net: Convolutional networks for biomedical image segmentation.
  MICCAI. https://arxiv.org/abs/1505.04597
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Channel configuration ──────────────────────────────────────────────────────
# These control the network capacity.  Doubling channels at each encoder level
# is the standard U-Net design.
ENCODER_CHANNELS: Tuple[int, ...] = (1, 64, 128, 256, 512)
BOTTLENECK_CHANNELS: int = 512
# Decoder channels (excluding skip-connection input) mirror the encoder in reverse
DECODER_CHANNELS: Tuple[int, ...] = (512, 256, 128, 64)
OUTPUT_CHANNELS: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Building-block modules
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNLeakyReLU(nn.Module):
    """
    A standard convolutional block: Conv2d → BatchNorm2d → LeakyReLU.

    Used in the encoder and bottleneck.  LeakyReLU (with negative slope 0.2)
    is preferred over plain ReLU in image synthesis models because it avoids
    the "dying ReLU" problem where neurons permanently output zero.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size:  Convolution kernel size (default 3).
        padding:      Zero-padding on each side (default 1 = "same" padding).
        negative_slope: Slope for negative values in LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            # bias=False because BatchNorm already has a learnable bias term
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, in_channels, H, W).
        Returns:
            Output tensor (B, out_channels, H, W).
        """
        return self.block(x)


class ConvBNReLU(nn.Module):
    """
    A standard convolutional block: Conv2d → BatchNorm2d → ReLU.

    Used in the decoder.  Plain ReLU is fine here since the decoder does
    not discriminate between real/fake — it just has to reconstruct.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, in_channels, H, W).
        Returns:
            Output tensor (B, out_channels, H, W).
        """
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    One encoder stage: ConvBNLeakyReLU + MaxPool2d.

    The convolution extracts features at the current resolution, then
    MaxPool halves the spatial dimensions (H, W → H/2, W/2).

    Args:
        in_channels:  Channels coming in.
        out_channels: Channels after convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBNLeakyReLU(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor (B, in_channels, H, W).

        Returns:
            Tuple of:
              - skip: Feature map BEFORE pooling — saved for the skip connection.
              - pooled: Feature map AFTER pooling (half the spatial size).
        """
        skip = self.conv(x)      # Keep at full resolution for skip connection
        pooled = self.pool(skip) # Downsample for the next encoder block
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    One decoder stage: Upsample → Concatenate skip → ConvBNReLU.

    Bilinear upsampling (rather than transposed convolutions) avoids the
    "checkerboard artifacts" that transposed convolutions can produce.
    After upsampling, the skip connection from the matching encoder level
    is concatenated, doubling the channel count before the convolution.

    Args:
        in_channels:   Channels from the previous decoder block.
        skip_channels: Channels in the corresponding encoder skip feature map.
        out_channels:  Channels after the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # After concatenating skip, channel count = in_channels + skip_channels
        self.conv = ConvBNReLU(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Args:
            x:    Feature map from previous decoder block (B, in_channels, H, W).
            skip: Matching encoder skip feature map (B, skip_channels, 2H, 2W).

        Returns:
            Output tensor (B, out_channels, 2H, 2W).
        """
        x = self.upsample(x)
        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for single-channel image-to-image translation (MRI → sCT).

    Architecture:
      Encoder:    4 blocks — channels 1 → 64 → 128 → 256 → 512
      Bottleneck: 1 block  — channels 512 → 512
      Decoder:    4 blocks — channels 512 → 256 → 128 → 64
      Head:       Conv 64 → 1, tanh

    The tanh activation ensures outputs are in [−1, 1], matching the
    normalized CT range used during training.

    Args:
        in_channels:  Number of input channels (1 for grayscale MRI).
        out_channels: Number of output channels (1 for grayscale CT).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each block: Conv+BN+LReLU → MaxPool
        # We override the first block's in_channels with the argument
        enc_ch = list(ENCODER_CHANNELS)
        enc_ch[0] = in_channels  # Allow flexible input channels

        self.enc1 = EncoderBlock(enc_ch[0], enc_ch[1])   # 256 → 128
        self.enc2 = EncoderBlock(enc_ch[1], enc_ch[2])   # 128 → 64
        self.enc3 = EncoderBlock(enc_ch[2], enc_ch[3])   # 64  → 32
        self.enc4 = EncoderBlock(enc_ch[3], enc_ch[4])   # 32  → 16

        # ── Bottleneck ───────────────────────────────────────────────────────
        # Deepest representation — no spatial change
        self.bottleneck = ConvBNLeakyReLU(enc_ch[4], BOTTLENECK_CHANNELS)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Each block: Upsample → Cat(skip) → Conv+BN+ReLU
        self.dec4 = DecoderBlock(BOTTLENECK_CHANNELS, enc_ch[4], DECODER_CHANNELS[0])
        self.dec3 = DecoderBlock(DECODER_CHANNELS[0], enc_ch[3], DECODER_CHANNELS[1])
        self.dec2 = DecoderBlock(DECODER_CHANNELS[1], enc_ch[2], DECODER_CHANNELS[2])
        self.dec1 = DecoderBlock(DECODER_CHANNELS[2], enc_ch[1], DECODER_CHANNELS[3])

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(DECODER_CHANNELS[3], out_channels, kernel_size=1),
            nn.Tanh(),  # Map to [−1, 1] to match CT normalization range
        )

        # Log parameter count (useful for reporting in portfolio README)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("UNet initialized — %d trainable parameters.", n_params)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: MRI slice → synthetic CT slice.

        Args:
            x: Input MRI tensor of shape (B, 1, 256, 256), values in [0, 1].

        Returns:
            Synthetic CT tensor of shape (B, 1, 256, 256), values in [−1, 1].
        """
        # ── Encode ───────────────────────────────────────────────────────────
        skip1, x = self.enc1(x)  # skip1: (B,  64, 256, 256), x: (B,  64, 128, 128)
        skip2, x = self.enc2(x)  # skip2: (B, 128, 128, 128), x: (B, 128,  64,  64)
        skip3, x = self.enc3(x)  # skip3: (B, 256,  64,  64), x: (B, 256,  32,  32)
        skip4, x = self.enc4(x)  # skip4: (B, 512,  32,  32), x: (B, 512,  16,  16)

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)   # x: (B, 512, 16, 16)

        # ── Decode ───────────────────────────────────────────────────────────
        x = self.dec4(x, skip4)  # x: (B, 512, 32, 32)
        x = self.dec3(x, skip3)  # x: (B, 256, 64, 64)
        x = self.dec2(x, skip2)  # x: (B, 128, 128, 128)
        x = self.dec1(x, skip1)  # x: (B,  64, 256, 256)

        # ── Output ───────────────────────────────────────────────────────────
        return self.head(x)      # x: (B, 1, 256, 256) ∈ [−1, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run: python src/model_unet.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    model = UNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)   # Batch of 2 MRI slices
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")   # torch.Size([2, 1, 256, 256])
    print(f"Output shape: {output.shape}")        # torch.Size([2, 1, 256, 256])
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # ≈ [−1, 1]
