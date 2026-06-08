"""
src/train_baseline.py
=====================
Training script for the baseline U-Net MRI → CT synthesis model.

Loss function: L1 (mean absolute error) between predicted and real CT.
L1 loss was chosen over L2 (MSE) because L1 produces sharper outputs —
L2 tends to create blurry predictions by averaging over multiple plausible
solutions (it penalizes outlier errors more heavily, encouraging the model
to predict the mean).

Usage (Windows PowerShell)
--------------------------
    python src/train_baseline.py --data_dir data/brain --epochs 50

All arguments have sensible defaults and can be overridden via command line.
Run with --help to see all options.

Output files
------------
  checkpoints/unet_epoch_{N}.pth     — saved model weights
  outputs/unet_training_log.csv      — per-epoch train + val losses
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Add src/ to path so we can import local modules without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloader
from model_unet import UNet
from utils import get_device, setup_logging

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Default hyperparameters (configurable via argparse) ───────────────────────
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LR: float = 2e-4           # Adam learning rate
DEFAULT_DATA_DIR: str = "data/brain"
DEFAULT_DEVICE: str = "cuda"
CHECKPOINT_EVERY: int = 10         # Save a checkpoint every N epochs
CHECKPOINT_DIR: Path = Path("checkpoints")
OUTPUT_DIR: Path = Path("outputs")
LOG_CSV_NAME: str = "unet_training_log.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with all training configuration fields.
    """
    parser = argparse.ArgumentParser(
        description="Train the baseline U-Net for MRI → CT synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS,     help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE, help="Batch size per GPU step.")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR,         help="Adam learning rate.")
    parser.add_argument("--data_dir",   type=str,   default=DEFAULT_DATA_DIR,   help="Root of the dataset directory.")
    parser.add_argument("--device",     type=str,   default=DEFAULT_DEVICE,     help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--num_workers",type=int,   default=4,                  help="DataLoader worker processes (use 0 on Windows if multiprocessing errors occur).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full training epoch over the DataLoader.

    Args:
        model:     U-Net model (on device).
        loader:    Training DataLoader.
        optimizer: Adam optimizer.
        criterion: L1 loss function.
        device:    torch.device to move batches to.

    Returns:
        Mean training loss for this epoch (float).
    """
    model.train()
    total_loss = 0.0

    for mri, real_ct in tqdm(loader, desc="  Train", leave=False, unit="batch"):
        # Move data to GPU (or CPU if no GPU available)
        mri = mri.to(device)
        real_ct = real_ct.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        pred_ct = model(mri)

        # L1 loss: encourages exact pixel-level accuracy
        loss = criterion(pred_ct, real_ct)

        # ── Backward pass ─────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()  # Disable gradient tracking during validation (saves memory)
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set without gradient computation.

    Args:
        model:     U-Net model (on device).
        loader:    Validation DataLoader.
        criterion: L1 loss function.
        device:    torch.device to move batches to.

    Returns:
        Mean validation loss for this epoch (float).
    """
    model.eval()
    total_loss = 0.0

    for mri, real_ct in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        mri = mri.to(device)
        real_ct = real_ct.to(device)

        pred_ct = model(mri)
        loss = criterion(pred_ct, real_ct)
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
) -> None:
    """
    Save model weights and optimizer state to a checkpoint file.

    We save the optimizer state as well so training can be resumed from
    a checkpoint with the same learning rate schedule and momentum.

    Args:
        model:          The U-Net model.
        optimizer:      Adam optimizer.
        epoch:          Current epoch number (used in filename).
        val_loss:       Validation loss at this epoch.
        checkpoint_dir: Directory to save checkpoint files.

    Returns:
        None
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"unet_epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Checkpoint saved: %s  (val_loss=%.6f)", path, val_loss)


def save_best_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
) -> None:
    """
    Save a 'best model' checkpoint (weights only, no optimizer state).

    This lightweight checkpoint is what the Gradio app will load for inference.

    Args:
        model:          The U-Net model at its best validation loss.
        checkpoint_dir: Directory to save the best checkpoint file.

    Returns:
        None
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "unet_best.pth"
    torch.save(model.state_dict(), path)
    logger.info("Best model saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

def init_csv_log(csv_path: Path) -> None:
    """
    Create the CSV log file and write the header row.

    Args:
        csv_path: Path to the output CSV file.

    Returns:
        None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])


def append_csv_log(csv_path: Path, epoch: int, train_loss: float, val_loss: float) -> None:
    """
    Append one epoch's results to the CSV log.

    Args:
        csv_path:   Path to the output CSV file.
        epoch:      Epoch number.
        train_loss: Mean training loss for this epoch.
        val_loss:   Mean validation loss for this epoch.

    Returns:
        None
    """
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])


def save_loss_plot(csv_path: Path, plot_path: Path) -> None:
    """
    Read training log CSV and plot loss curves using standard csv reader.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Prevent GUI window generation
        import matplotlib.pyplot as plt
        
        epochs = []
        g_loss = []
        d_loss = []
        val_l1 = []
        train_l1 = []
        
        is_pix2pix = False
        
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "G_loss" in header:
                is_pix2pix = True
            
            for row in reader:
                if not row:
                    continue
                epochs.append(int(row[0]))
                if is_pix2pix:
                    g_loss.append(float(row[1]))
                    d_loss.append(float(row[2]))
                    val_l1.append(float(row[3]))
                else:
                    train_l1.append(float(row[1]))
                    val_l1.append(float(row[2]))
                    
        fig, ax = plt.subplots(figsize=(10, 5))
        if is_pix2pix:
            ax.plot(epochs, g_loss, label="Generator Loss (BCE + L1)", color="blue")
            ax.plot(epochs, d_loss, label="Discriminator Loss (BCE)", color="red")
            ax.plot(epochs, val_l1, label="Val L1 Loss (Reconstruction)", color="green")
        else:
            ax.plot(epochs, train_l1, label="Train L1 Loss", color="blue")
            ax.plot(epochs, val_l1, label="Val L1 Loss", color="green")
            
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Curves")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning("Could not generate loss plot: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main training function.

    Orchestrates data loading, model initialization, training loop,
    checkpointing, and CSV logging for the baseline U-Net.

    Returns:
        None
    """
    setup_logging()
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("--device=cuda requested but CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Training device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading datasets from: %s", args.data_dir)
    train_loader = get_dataloader(
        args.data_dir, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        args.data_dir, split="val",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=1).to(device)

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.L1Loss()  # Mean absolute error — produces sharper images than MSE
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # betas=(0.5, 0.999) is the standard pix2pix setting; stabilizes GAN-style training

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / LOG_CSV_NAME
    init_csv_log(csv_path)
    logger.info("Training log will be written to: %s", csv_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d / %d", epoch, args.epochs)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Write to CSV log every epoch
        append_csv_log(csv_path, epoch, train_loss, val_loss)
        save_loss_plot(csv_path, OUTPUT_DIR / "unet_loss_curve.png")

        # Print human-readable summary to stdout (visible in terminal)
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"Train L1: {train_loss:.5f}  |  Val L1: {val_loss:.5f}"
        )

        # Save periodic checkpoint every CHECKPOINT_EVERY epochs
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, CHECKPOINT_DIR)

        # Track best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(model, CHECKPOINT_DIR)
            logger.info("New best model at epoch %d (val_loss=%.6f).", epoch, val_loss)

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"  Best weights saved to: {CHECKPOINT_DIR / 'unet_best.pth'}")
    print(f"  Training log saved to: {csv_path}")


if __name__ == "__main__":
    main()
