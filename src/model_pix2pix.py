"""
src/model_pix2pix.py
====================
pix2pix conditional GAN for MRI → CT image synthesis.

What is pix2pix?
----------------
pix2pix (Isola et al., 2017) is a conditional generative adversarial network
(cGAN) for image-to-image translation.  It extends a basic GAN by conditioning
both the generator and discriminator on the source image (MRI), so the model
learns to produce outputs that are consistent with the input — not just
"realistic-looking" CTs in general.

Two-player minimax game:
  - Generator G:  Given MRI, produce a synthetic CT that fools D.
  - Discriminator D: Given (MRI + CT) pair, decide if the CT is real or fake.
  D sees the MRI as context, making it a "conditional" discriminator.

Components
----------
  Generator:     Same U-Net as model_unet.py (encoder–decoder + skip connections).
                 Takes MRI as input, outputs synthetic CT.
  Discriminator: PatchGAN — instead of a single real/fake score for the whole
                 image, it outputs a spatial grid of scores.  Each output pixel
                 corresponds to a receptive field patch in the input.
                 This encourages the generator to produce locally sharp textures
                 rather than blurry global averages.

PatchGAN receptive field
------------------------
With 4 conv layers (stride=2 except last), the effective receptive field of
each output "patch" is ~70×70 pixels in the input.  This is a classic design
from the pix2pix paper.  Discriminating at the patch level is more efficient
than at the full-image level for high-resolution images.

Reference
---------
  Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017).
  Image-to-image translation with conditional adversarial networks.
  CVPR. https://arxiv.org/abs/1611.07004
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Re-use the U-Net architecture for the generator
from model_unet import UNet

logger = logging.getLogger(__name__)

# ── Discriminator architecture constants ──────────────────────────────────────
# Input has 2 channels: (MRI concatenated with CT)
DISC_IN_CHANNELS: int = 2       # MRI (1-ch) + CT (1-ch)
DISC_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512)  # Progressive channel widths
LEAKY_RELU_SLOPE: float = 0.2   # Standard slope for LeakyReLU in discriminators


# ─────────────────────────────────────────────────────────────────────────────
# PatchGAN Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies image patches as real or fake.

    Architecture (4-layer convolutional network):
      Layer 1: Conv(2→64,   stride=2) + LeakyReLU          — no BatchNorm (first layer)
      Layer 2: Conv(64→128, stride=2) + BatchNorm + LReLU
      Layer 3: Conv(128→256,stride=2) + BatchNorm + LReLU
      Layer 4: Conv(256→512,stride=1) + BatchNorm + LReLU
      Output:  Conv(512→1,  stride=1) + Sigmoid

    Input shape:  (B, 2, 256, 256) — MRI and CT (real or fake) concatenated
    Output shape: (B, 1, H', W')   — spatial grid of [0,1] real/fake scores

    The discriminator does NOT classify the full image; it classifies
    overlapping patches via its limited receptive field.  This is why we
    use spatial loss (BCE per pixel on the output grid) rather than a
    single sigmoid for the full image.

    Args:
        in_channels:  Number of input channels (2: MRI + CT).
    """

    def __init__(self, in_channels: int = DISC_IN_CHANNELS) -> None:
        super().__init__()

        # Build layers sequentially
        layers: list[nn.Module] = []

        ch_in = in_channels
        for i, ch_out in enumerate(DISC_CHANNELS):
            # Use stride=2 for all layers except the last (which uses stride=1
            # to preserve spatial resolution for the final output map)
            stride = 2 if i < len(DISC_CHANNELS) - 1 else 1
            layers.append(
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=stride, padding=1)
            )
            # BatchNorm is NOT applied to the very first layer — common practice
            # because the first layer acts directly on raw pixel values
            if i > 0:
                layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True))
            ch_in = ch_out

        # Final projection to a single real/fake channel
        layers.append(
            nn.Conv2d(ch_in, 1, kernel_size=4, stride=1, padding=1)
        )
        # Sigmoid maps raw logits → [0, 1] for BCELoss
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("PatchGANDiscriminator initialized — %d trainable parameters.", n_params)

    def forward(self, mri: Tensor, ct: Tensor) -> Tensor:
        """
        Forward pass: classify patch-wise whether (mri, ct) pair is real or fake.

        The MRI is concatenated with the CT along the channel dimension so the
        discriminator sees them as a 2-channel image pair.  This conditions the
        discriminator on the MRI input, which is the key "conditional" part
        of conditional GANs.

        Args:
            mri: MRI input tensor, shape (B, 1, H, W), values in [0, 1].
            ct:  CT tensor (real or generated), shape (B, 1, H, W), values in [−1, 1].

        Returns:
            Score map tensor (B, 1, H', W') with values in [0, 1].
            Higher values indicate the discriminator believes the pair is real.
        """
        # Concatenate MRI and CT along channel axis → (B, 2, H, W)
        pair = torch.cat([mri, ct], dim=1)
        return self.model(pair)


# ─────────────────────────────────────────────────────────────────────────────
# Pix2Pix container (Generator + Discriminator)
# ─────────────────────────────────────────────────────────────────────────────

class Pix2Pix(nn.Module):
    """
    Container that holds the pix2pix Generator and Discriminator together.

    The generator and discriminator are trained with SEPARATE optimizers
    (see train_pix2pix.py), so this class does NOT implement a single
    forward() that trains both.  Instead it exposes:
      - ``self.generator``      — U-Net (MRI → sCT)
      - ``self.discriminator``  — PatchGAN (MRI, CT → score map)

    This design makes it easy to:
      - Save/load only the generator for inference.
      - Freeze the discriminator during generator updates (and vice versa).
      - Use the generator standalone in the Gradio app.

    Args:
        in_channels:  Channels for the generator input (1 for MRI).
        out_channels: Channels for the generator output (1 for CT).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.generator = UNet(in_channels=in_channels, out_channels=out_channels)
        self.discriminator = PatchGANDiscriminator(
            in_channels=in_channels + out_channels  # MRI (1) + CT (1) = 2
        )

        logger.info("Pix2Pix model initialized.")

    def forward(self, mri: Tensor) -> Tensor:
        """
        Run only the generator (used for inference, not training).

        During training, the generator and discriminator are called
        separately with explicit loss computations in train_pix2pix.py.

        Args:
            mri: MRI input tensor (B, 1, H, W).

        Returns:
            Synthetic CT tensor (B, 1, H, W) ∈ [−1, 1].
        """
        return self.generator(mri)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run: python src/model_pix2pix.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    model = Pix2Pix()

    mri_batch = torch.randn(2, 1, 256, 256)    # Batch of 2 MRI slices
    ct_batch = torch.randn(2, 1, 256, 256)     # Batch of 2 real CT slices

    # Generator: MRI → synthetic CT
    fake_ct = model.generator(mri_batch)
    print(f"Generator output: {fake_ct.shape}")  # (2, 1, 256, 256)

    # Discriminator: (MRI, real CT) → patch scores
    real_scores = model.discriminator(mri_batch, ct_batch)
    print(f"Discriminator output (real): {real_scores.shape}")  # (2, 1, H', W')

    # Discriminator: (MRI, fake CT) → patch scores
    fake_scores = model.discriminator(mri_batch, fake_ct.detach())
    print(f"Discriminator output (fake): {fake_scores.shape}")  # (2, 1, H', W')
    print(f"Fake CT range: [{fake_ct.min():.3f}, {fake_ct.max():.3f}]")
