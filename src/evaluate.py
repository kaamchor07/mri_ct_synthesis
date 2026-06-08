"""
src/evaluate.py
===============
Evaluation script for trained MRI → CT synthesis models.

Metrics computed on the test set
---------------------------------
  MAE   — Mean Absolute Error in Hounsfield Units (HU).
           Lower is better.  Clinical target: <40 HU for brain.
  PSNR  — Peak Signal-to-Noise Ratio (dB).
           Higher is better.  Measures pixel-level fidelity.
  SSIM  — Structural Similarity Index Measure (range [0, 1]).
           Higher is better.  Measures perceptual / structural quality.

Output files
------------
  outputs/eval_grid.png       — 5 random test cases: MRI | Pred CT | Real CT | Error
  outputs/failure_cases.png   — 3 worst SSIM cases (same layout)

Usage
-----
    # Evaluate the pix2pix generator
    python src/evaluate.py --checkpoint checkpoints/pix2pix_best.pth --model pix2pix

    # Evaluate the U-Net baseline
    python src/evaluate.py --checkpoint checkpoints/unet_best.pth --model unet
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloader
from model_unet import UNet
from model_pix2pix import Pix2Pix
from utils import denormalize_ct, setup_logging

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR: str = "data/brain"
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_DEVICE: str = "cuda"
OUTPUT_DIR: Path = Path("outputs")
N_DISPLAY_CASES: int = 5     # Number of cases in eval_grid.png
N_FAILURE_CASES: int = 3     # Number of worst-SSIM cases in failure_cases.png


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace with checkpoint path, model type, data dir, etc.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MRI→CT model on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the saved model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--model", type=str, default="pix2pix", choices=["unet", "pix2pix"],
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help="Root of the dataset directory.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help="Device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint file.

    For pix2pix checkpoints, we load only the generator weights — the
    discriminator is not needed for inference.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model_type:      "unet" or "pix2pix".
        device:          torch.device to load the model onto.

    Returns:
        The loaded model in eval mode, on the specified device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError:        If model_type is invalid.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model: torch.nn.Module
    checkpoint = torch.load(path, map_location=device)

    if model_type == "unet":
        model = UNet(in_channels=1, out_channels=1)
        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    elif model_type == "pix2pix":
        # For pix2pix, we only need the generator at inference time
        model = UNet(in_channels=1, out_channels=1)
        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["generator_state_dict"])
        else:
            # Assume the checkpoint is just the generator state dict
            model.load_state_dict(checkpoint)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'unet' or 'pix2pix'.")

    model = model.to(device)
    model.eval()  # Switch to eval mode: disables BatchNorm updates and Dropout
    logger.info("Loaded %s model from %s", model_type, path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_slice(
    pred_ct_norm: np.ndarray,
    real_ct_norm: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute MAE (HU), PSNR, and SSIM for a single predicted CT slice.

    Both inputs should be 2D arrays normalized to [−1, 1].  We de-normalize
    to HU before computing MAE (for clinical interpretability), but use the
    normalized versions for PSNR and SSIM (which expect a fixed data_range).

    Args:
        pred_ct_norm: Predicted CT, shape (H, W), values in [−1, 1].
        real_ct_norm: Real CT,      shape (H, W), values in [−1, 1].

    Returns:
        Tuple of (mae_hu, psnr_db, ssim_val).
    """
    # MAE in HU — de-normalize first for physical interpretability
    pred_hu = denormalize_ct(pred_ct_norm)
    real_hu = denormalize_ct(real_ct_norm)
    mae = float(np.mean(np.abs(pred_hu - real_hu)))

    # PSNR uses the normalized range [−1, 1], so data_range = 2.0
    psnr = float(psnr_fn(real_ct_norm, pred_ct_norm, data_range=2.0))

    # SSIM: also use normalized range; channel_axis=None for 2D arrays
    ssim = float(ssim_fn(real_ct_norm, pred_ct_norm, data_range=2.0))

    return mae, psnr, ssim


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_grid(
    cases: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]],
    save_path: Path,
    title: str = "Evaluation Grid",
) -> None:
    """
    Save a multi-row figure showing [MRI | Pred CT | Real CT | Abs Error].

    Args:
        cases: List of (mri, pred_ct, real_ct, mae, psnr, ssim) tuples.
               All arrays should be 2D (H, W).
        save_path: Output PNG path.
        title:     Figure super-title.

    Returns:
        None
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))

    # Handle n=1 case where axes is 1D
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI Input", "Predicted sCT", "Real CT", "Abs Error (HU)"]
    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontsize=13, fontweight="bold", pad=8)

    for row, (mri, pred, real, mae, psnr, ssim) in enumerate(cases):
        pred_hu = denormalize_ct(pred)
        real_hu = denormalize_ct(real)
        abs_err = np.abs(pred_hu - real_hu)

        axes[row, 0].imshow(mri, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred, cmap="gray", vmin=-1, vmax=1)
        axes[row, 1].set_ylabel(
            f"MAE={mae:.1f} HU\nPSNR={psnr:.1f} dB\nSSIM={ssim:.3f}",
            fontsize=9, rotation=0, labelpad=55, va="center",
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(real, cmap="gray", vmin=-1, vmax=1)
        axes[row, 2].axis("off")

        im = axes[row, 3].imshow(abs_err, cmap="hot", vmin=0, vmax=300)
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=15, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved eval figure → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main() -> None:
    """
    Main evaluation function.

    Loads a checkpoint, runs inference on the test set, computes MAE/PSNR/SSIM
    for every slice, prints summary statistics, and saves evaluation figures.

    Returns:
        None
    """
    setup_logging()
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, args.model, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    test_loader = get_dataloader(
        args.data_dir, split="test",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    logger.info("Running inference on %d test batches...", len(test_loader))

    # ── Collect per-slice results ─────────────────────────────────────────────
    all_mae:  List[float] = []
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    # Store (mri, pred_ct, real_ct, mae, psnr, ssim) for visualization
    all_cases: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]] = []

    for mri_batch, real_ct_batch in test_loader:
        mri_batch = mri_batch.to(device)
        pred_ct_batch = model(mri_batch)

        # Process each slice in the batch individually
        for i in range(mri_batch.size(0)):
            mri_np = mri_batch[i, 0].cpu().numpy()
            pred_np = pred_ct_batch[i, 0].cpu().numpy()
            real_np = real_ct_batch[i, 0].numpy()

            mae, psnr, ssim = compute_metrics_for_slice(pred_np, real_np)

            all_mae.append(mae)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_cases.append((mri_np, pred_np, real_np, mae, psnr, ssim))

    # ── Summary statistics ────────────────────────────────────────────────────
    mae_arr  = np.array(all_mae)
    psnr_arr = np.array(all_psnr)
    ssim_arr = np.array(all_ssim)

    print("\n" + "=" * 55)
    print(f"  Model:   {args.model}  |  Checkpoint: {args.checkpoint}")
    print(f"  N slices tested: {len(all_cases)}")
    print("=" * 55)
    print(f"  MAE  (HU):  {mae_arr.mean():.2f} ± {mae_arr.std():.2f}")
    print(f"  PSNR (dB):  {psnr_arr.mean():.2f} ± {psnr_arr.std():.2f}")
    print(f"  SSIM:       {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}")
    print("=" * 55 + "\n")

    # ── Evaluation grid (random sample) ──────────────────────────────────────
    random.seed(42)
    sample_indices = random.sample(range(len(all_cases)), min(N_DISPLAY_CASES, len(all_cases)))
    sample_cases = [all_cases[i] for i in sample_indices]

    make_eval_grid(
        sample_cases,
        save_path=OUTPUT_DIR / "eval_grid.png",
        title=f"Evaluation Grid — {args.model} (N={len(all_cases)} test slices)",
    )

    # ── Failure cases (worst SSIM) ────────────────────────────────────────────
    # Sort by SSIM ascending — worst cases first
    sorted_cases = sorted(all_cases, key=lambda c: c[5])  # c[5] = ssim value
    failure_cases = sorted_cases[:N_FAILURE_CASES]

    make_eval_grid(
        failure_cases,
        save_path=OUTPUT_DIR / "failure_cases.png",
        title=f"Failure Cases (Worst SSIM) — {args.model}",
    )

    print(f"✓ Eval grid saved    → {OUTPUT_DIR / 'eval_grid.png'}")
    print(f"✓ Failure cases saved → {OUTPUT_DIR / 'failure_cases.png'}")


if __name__ == "__main__":
    main()
