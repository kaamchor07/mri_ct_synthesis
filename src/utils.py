"""
src/utils.py
============
Shared utility functions used across training, evaluation, and the Gradio app.

Responsibilities
----------------
- MRI normalization (percentile clip → [0, 1])
- CT normalization (HU clip → [−1, 1]) and de-normalization
- Saving comparison figures (MRI | Predicted CT | Real CT)
- Device detection helper

All functions are stateless and side-effect-free (except save_comparison_figure,
which writes a file).
"""

import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Normalization constants ────────────────────────────────────────────────────
# CT Hounsfield Unit clipping range (air ≈ −1000 HU, dense bone ≈ +1000 HU)
CT_HU_MIN: float = -1000.0
CT_HU_MAX: float = 1000.0

# MRI percentile clip (removes extreme outliers from scanner noise/artifacts)
MRI_PERCENTILE_LOW: float = 1.0
MRI_PERCENTILE_HIGH: float = 99.0


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Normalize an MRI volume to the range [0, 1].

    Uses percentile clipping to remove scanner-specific outliers
    (e.g. bright noise at image borders) before scaling.

    Args:
        volume: Raw MRI numpy array of arbitrary shape and dtype.

    Returns:
        Float32 numpy array with values clipped and scaled to [0, 1].
    """
    volume = volume.astype(np.float32)

    # Compute robust min/max via percentiles — ignores extreme outliers
    p_low = np.percentile(volume, MRI_PERCENTILE_LOW)
    p_high = np.percentile(volume, MRI_PERCENTILE_HIGH)

    logger.debug(
        "MRI normalization: p%.0f=%.2f, p%.0f=%.2f",
        MRI_PERCENTILE_LOW, p_low,
        MRI_PERCENTILE_HIGH, p_high,
    )

    # Clip, then scale to [0, 1]
    volume = np.clip(volume, p_low, p_high)

    # Avoid division by zero for flat (all-zero) volumes
    denom = p_high - p_low
    if denom < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - p_low) / denom
    return volume.astype(np.float32)


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """
    Normalize a CT volume from Hounsfield Units (HU) to [−1, 1].

    The clipping range [−1000, +1000] HU covers all clinically relevant
    soft-tissue contrast while excluding extreme implant/air artifacts.
    After clipping, values are linearly mapped to [−1, 1] to match the
    tanh output range of both the U-Net and pix2pix generator.

    Args:
        volume: Raw CT numpy array in Hounsfield Units.

    Returns:
        Float32 numpy array with values in [−1, 1].
    """
    volume = volume.astype(np.float32)

    # Hard-clip to clinically relevant HU range
    volume = np.clip(volume, CT_HU_MIN, CT_HU_MAX)

    # Linear map: [CT_HU_MIN, CT_HU_MAX] → [−1, 1]
    # Formula: y = (x − min) / (max − min) * 2 − 1
    volume = (volume - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN) * 2.0 - 1.0
    return volume.astype(np.float32)


def denormalize_ct(tensor: Union[Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a normalized CT tensor/array back to Hounsfield Units.

    Inverts normalize_ct: maps [−1, 1] → [−1000, 1000] HU.
    This is required before computing MAE in HU for evaluation.

    Args:
        tensor: A torch.Tensor or numpy array with values in [−1, 1].
                Can have any shape (e.g. (1, H, W) or (H, W)).

    Returns:
        Float32 numpy array in Hounsfield Units [−1000, 1000].
    """
    if isinstance(tensor, Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    # Inverse of the linear map: y = (x + 1) / 2 * (max − min) + min
    hu = (arr + 1.0) / 2.0 * (CT_HU_MAX - CT_HU_MIN) + CT_HU_MIN
    return hu.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_figure(
    mri: np.ndarray,
    pred_ct: np.ndarray,
    real_ct: np.ndarray,
    save_path: Union[str, Path],
    title: str = "",
) -> None:
    """
    Save a three-panel comparison figure: MRI | Predicted CT | Real CT.

    All inputs should be 2D arrays (H, W). If a leading channel dimension
    is present (1, H, W), it will be squeezed automatically.

    Args:
        mri:        Normalized MRI slice, values in [0, 1].
        pred_ct:    Predicted CT slice, values in [−1, 1].
        real_ct:    Ground-truth CT slice, values in [−1, 1].
        save_path:  File path to save the PNG figure.
        title:      Optional super-title for the figure.

    Returns:
        None  (writes PNG file as side effect).
    """
    # Squeeze channel dimension if present (e.g. shape (1,256,256) → (256,256))
    mri = np.squeeze(mri)
    pred_ct = np.squeeze(pred_ct)
    real_ct = np.squeeze(real_ct)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mri, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("MRI Input")
    axes[0].axis("off")

    axes[1].imshow(pred_ct, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Predicted sCT")
    axes[1].axis("off")

    axes[2].imshow(real_ct, cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title("Real CT")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved comparison figure → %s", save_path)


def save_quad_figure(
    mri: np.ndarray,
    pred_ct: np.ndarray,
    real_ct: np.ndarray,
    save_path: Union[str, Path],
    title: str = "",
) -> None:
    """
    Save a four-panel figure: MRI | Predicted CT | Real CT | Absolute Error.

    The absolute error map uses a hot colormap to make large errors visible.
    Both CT images are de-normalized to HU before computing the error so
    the scale is clinically interpretable (in HU).

    Args:
        mri:        Normalized MRI slice, values in [0, 1]. Shape (H, W).
        pred_ct:    Predicted CT slice, values in [−1, 1]. Shape (H, W).
        real_ct:    Ground-truth CT slice, values in [−1, 1]. Shape (H, W).
        save_path:  File path to save the PNG figure.
        title:      Optional super-title for the figure.

    Returns:
        None  (writes PNG file as side effect).
    """
    mri = np.squeeze(mri)
    pred_ct_arr = np.squeeze(pred_ct)
    real_ct_arr = np.squeeze(real_ct)

    # Compute absolute error in HU for interpretability
    pred_hu = denormalize_ct(pred_ct_arr)
    real_hu = denormalize_ct(real_ct_arr)
    abs_error = np.abs(pred_hu - real_hu)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(mri, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("MRI Input")
    axes[0].axis("off")

    axes[1].imshow(pred_ct_arr, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Predicted sCT")
    axes[1].axis("off")

    axes[2].imshow(real_ct_arr, cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title("Real CT")
    axes[2].axis("off")

    im = axes[3].imshow(abs_error, cmap="hot", vmin=0, vmax=200)
    axes[3].set_title("Abs. Error (HU)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved quad figure → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Prefers CUDA (NVIDIA GPU) for training speed, falls back to CPU.
    Logs which device was selected so the user knows what hardware is active.

    Returns:
        torch.device — either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Using GPU: %s", gpu_name)
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available — running on CPU. "
            "Training will be significantly slower."
        )
    return device


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a consistent format for all modules.

    Call this once at the top of any training/evaluation script.

    Args:
        level: Logging verbosity level (default: logging.INFO).

    Returns:
        None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
