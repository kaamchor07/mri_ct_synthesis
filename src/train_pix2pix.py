"""
src/train_pix2pix.py
====================
Training script for the pix2pix conditional GAN (MRI → CT synthesis).

Loss formulation
----------------
pix2pix uses a combined loss for the generator:

  L_G = lambda_L1 * L1(G(mri), real_ct)   ← pixel-level accuracy
       + BCE(D(mri, G(mri)), ones)          ← adversarial fooling of D

  L_D = BCE(D(mri, real_ct), ones)         ← D should classify real as real
       + BCE(D(mri, G(mri).detach()), zeros) ← D should classify fake as fake

Key notes:
  - lambda_L1 = 100 weights the L1 term heavily — this keeps the generated
    CT structurally close to the real CT while the GAN component sharpens details.
  - We use BCE (binary cross-entropy) with Sigmoid output from the discriminator.
  - `G(mri).detach()` is critical during the discriminator update step.
    Detaching stops gradients from flowing back into the generator — we want
    to improve D independently, without accidentally updating G.
  - Separate optimizers are used for G and D so they can be updated
    independently in each training step.

Training order per batch:
  1. Forward pass through G → fake_ct
  2. Update D: compute D_loss using (real_ct) and (fake_ct.detach())
  3. Update G: compute G_loss using D(mri, fake_ct) (without detach this time)

Usage
-----
    python src/train_pix2pix.py --data_dir data/brain --epochs 50

Output files
------------
  checkpoints/pix2pix_epoch_{N}.pth   — generator + discriminator weights
  checkpoints/pix2pix_best.pth        — best generator weights (for Gradio)
  outputs/pix2pix_training_log.csv    — per-epoch G_loss and D_loss
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

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloader
from model_pix2pix import Pix2Pix
from utils import get_device, setup_logging

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LR: float = 2e-4
DEFAULT_LAMBDA_L1: float = 100.0   # Weight of pixel-level L1 loss in G loss
DEFAULT_DATA_DIR: str = "data/brain"
DEFAULT_DEVICE: str = "cuda"
CHECKPOINT_EVERY: int = 10
CHECKPOINT_DIR: Path = Path("checkpoints")
OUTPUT_DIR: Path = Path("outputs")
LOG_CSV_NAME: str = "pix2pix_training_log.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for pix2pix training.

    Returns:
        argparse.Namespace with all training configuration fields.
    """
    parser = argparse.ArgumentParser(
        description="Train pix2pix cGAN for MRI → CT synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS,     help="Number of training epochs.")
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE, help="Batch size per GPU step.")
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR,         help="Adam learning rate for both G and D.")
    parser.add_argument("--lambda_l1",   type=float, default=DEFAULT_LAMBDA_L1,  help="Weight of L1 loss in generator loss.")
    parser.add_argument("--data_dir",    type=str,   default=DEFAULT_DATA_DIR,   help="Root of the dataset directory.")
    parser.add_argument("--device",      type=str,   default=DEFAULT_DEVICE,     help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--num_workers", type=int,   default=4,                  help="DataLoader worker processes.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Target tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def ones_like(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Create a tensor of ones matching the shape of a discriminator output.

    Used as the "real" target labels for BCE loss.

    Args:
        tensor: Discriminator output tensor (shape determines size).
        device: Target device.

    Returns:
        Tensor of ones with the same shape as `tensor`, on `device`.
    """
    return torch.ones_like(tensor, device=device)


def zeros_like(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Create a tensor of zeros matching the shape of a discriminator output.

    Used as the "fake" target labels for BCE loss.

    Args:
        tensor: Discriminator output tensor (shape determines size).
        device: Target device.

    Returns:
        Tensor of zeros with the same shape as `tensor`, on `device`.
    """
    return torch.zeros_like(tensor, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: Pix2Pix,
    loader: torch.utils.data.DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    criterion_GAN: nn.Module,
    criterion_L1: nn.Module,
    lambda_l1: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full training epoch for both generator and discriminator.

    Order of operations per batch:
      1. Generate fake CT: fake_ct = G(mri)
      2. Update D:
           D_loss = 0.5 * [BCE(D(mri,real_ct), 1) + BCE(D(mri,fake_ct.detach()), 0)]
      3. Update G:
           G_loss = BCE(D(mri,fake_ct), 1) + lambda_l1 * L1(fake_ct, real_ct)

    Args:
        model:        Pix2Pix model containing generator and discriminator.
        loader:       Training DataLoader.
        opt_G:        Adam optimizer for the generator.
        opt_D:        Adam optimizer for the discriminator.
        criterion_GAN: BCE loss for adversarial objectives.
        criterion_L1:  L1 loss for pixel-level accuracy.
        lambda_l1:    Weight of the L1 loss term.
        device:       torch.device for tensor placement.

    Returns:
        Tuple of (mean_G_loss, mean_D_loss) for this epoch.
    """
    model.train()
    total_G_loss = 0.0
    total_D_loss = 0.0

    for mri, real_ct in tqdm(loader, desc="  Train", leave=False, unit="batch"):
        mri = mri.to(device)
        real_ct = real_ct.to(device)

        # ─── Step 1: Generate fake CT ──────────────────────────────────────
        fake_ct = model.generator(mri)  # (B, 1, H, W)

        # ─── Step 2: Update Discriminator ─────────────────────────────────
        # We train D to correctly classify real and fake pairs.
        # IMPORTANT: We call fake_ct.detach() here to prevent gradients from
        # flowing back through the generator.  The discriminator update should
        # only improve D — we don't want it to accidentally degrade G.
        opt_D.zero_grad()

        # D on real pair (mri, real_ct) — should output 1 (real)
        pred_real = model.discriminator(mri, real_ct)
        loss_D_real = criterion_GAN(pred_real, ones_like(pred_real, device))

        # D on fake pair (mri, fake_ct) — should output 0 (fake)
        # Use detach() so gradients stop at fake_ct and don't reach G
        pred_fake = model.discriminator(mri, fake_ct.detach())
        loss_D_fake = criterion_GAN(pred_fake, zeros_like(pred_fake, device))

        # Total D loss: average of real and fake losses
        # Dividing by 2 slows D down relative to G — a common stabilization trick
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # ─── Step 3: Update Generator ──────────────────────────────────────
        # We train G to fool D and to match the real CT at the pixel level.
        # NOTE: this time we do NOT detach fake_ct — we want gradients to flow
        # all the way back through G so it can learn from the adversarial signal.
        opt_G.zero_grad()

        # Adversarial loss: G wants D to output 1 (real) for fake pairs
        pred_fake_for_G = model.discriminator(mri, fake_ct)
        loss_G_GAN = criterion_GAN(pred_fake_for_G, ones_like(pred_fake_for_G, device))

        # Pixel-level L1 loss: G should also match the real CT closely
        loss_G_L1 = criterion_L1(fake_ct, real_ct)

        # Combined generator loss
        loss_G = loss_G_GAN + lambda_l1 * loss_G_L1
        loss_G.backward()
        opt_G.step()

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()

    n = len(loader)
    return total_G_loss / n, total_D_loss / n


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop (generator only, using L1 loss as a comparable metric)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: Pix2Pix,
    loader: torch.utils.data.DataLoader,
    criterion_L1: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate generator quality on the validation set using L1 loss.

    We only measure L1 here (not GAN loss) because the discriminator is
    not meaningful in eval mode and L1 is a stable, interpretable metric.

    Args:
        model:        Pix2Pix model.
        loader:       Validation DataLoader.
        criterion_L1: L1 loss function.
        device:       torch.device.

    Returns:
        Mean L1 loss across all validation batches.
    """
    model.eval()
    total_loss = 0.0

    for mri, real_ct in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        mri = mri.to(device)
        real_ct = real_ct.to(device)
        fake_ct = model.generator(mri)
        loss = criterion_L1(fake_ct, real_ct)
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: Pix2Pix,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
) -> None:
    """
    Save both generator and discriminator state to a checkpoint file.

    Args:
        model:          Pix2Pix model.
        opt_G:          Generator optimizer.
        opt_D:          Discriminator optimizer.
        epoch:          Current epoch number.
        val_loss:       Validation L1 loss at this epoch.
        checkpoint_dir: Directory to save checkpoint files.

    Returns:
        None
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"pix2pix_epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": model.generator.state_dict(),
            "discriminator_state_dict": model.discriminator.state_dict(),
            "opt_G_state_dict": opt_G.state_dict(),
            "opt_D_state_dict": opt_D.state_dict(),
            "val_l1_loss": val_loss,
        },
        path,
    )
    logger.info("Checkpoint saved: %s  (val_L1=%.6f)", path, val_loss)


def save_best_generator(model: Pix2Pix, checkpoint_dir: Path) -> None:
    """
    Save only the generator's weights as the 'best' checkpoint.

    This lightweight file is loaded by the Gradio app for inference.

    Args:
        model:          Pix2Pix model.
        checkpoint_dir: Directory to save.

    Returns:
        None
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "pix2pix_best.pth"
    torch.save(model.generator.state_dict(), path)
    logger.info("Best generator saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

def init_csv_log(csv_path: Path) -> None:
    """
    Create the CSV training log file and write the header row.

    Args:
        csv_path: Path to the output CSV file.

    Returns:
        None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "G_loss", "D_loss", "val_L1_loss"])


def append_csv_log(
    csv_path: Path,
    epoch: int,
    g_loss: float,
    d_loss: float,
    val_loss: float,
) -> None:
    """
    Append one epoch's losses to the CSV log.

    Args:
        csv_path:  Path to the CSV file.
        epoch:     Epoch number.
        g_loss:    Mean generator loss.
        d_loss:    Mean discriminator loss.
        val_loss:  Mean validation L1 loss.

    Returns:
        None
    """
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{g_loss:.6f}", f"{d_loss:.6f}", f"{val_loss:.6f}"])


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
    Main training function for pix2pix.

    Orchestrates data loading, model initialization, alternating G/D training,
    checkpointing, and CSV logging.

    Returns:
        None
    """
    setup_logging()
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Training device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading datasets from: %s", args.data_dir)
    train_loader = get_dataloader(
        args.data_dir, "train", args.batch_size, args.num_workers
    )
    val_loader = get_dataloader(
        args.data_dir, "val", args.batch_size, args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Pix2Pix(in_channels=1, out_channels=1).to(device)

    # ── Loss functions ────────────────────────────────────────────────────────
    criterion_GAN = nn.BCELoss()   # Adversarial loss (discriminator uses Sigmoid)
    criterion_L1 = nn.L1Loss()     # Pixel-level reconstruction loss

    # ── Optimizers ────────────────────────────────────────────────────────────
    # betas=(0.5, 0.999): standard for GANs — lower beta1 reduces momentum,
    # which helps stabilize the adversarial training dynamic
    opt_G = Adam(model.generator.parameters(),     lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / LOG_CSV_NAME
    init_csv_log(csv_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d / %d", epoch, args.epochs)

        g_loss, d_loss = train_one_epoch(
            model, train_loader, opt_G, opt_D,
            criterion_GAN, criterion_L1, args.lambda_l1, device,
        )
        val_loss = validate(model, val_loader, criterion_L1, device)

        append_csv_log(csv_path, epoch, g_loss, d_loss, val_loss)
        save_loss_plot(csv_path, OUTPUT_DIR / "pix2pix_loss_curve.png")

        # Human-readable epoch summary
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"G_loss: {g_loss:.5f}  |  D_loss: {d_loss:.5f}  |  Val L1: {val_loss:.5f}"
        )

        # Periodic checkpoint (both G and D)
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt_G, opt_D, epoch, val_loss, CHECKPOINT_DIR)

        # Best generator (based on val L1 loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_generator(model, CHECKPOINT_DIR)
            logger.info("New best generator at epoch %d (val_L1=%.6f).", epoch, val_loss)

    print(f"\n✓ Training complete. Best val L1: {best_val_loss:.6f}")
    print(f"  Best generator: {CHECKPOINT_DIR / 'pix2pix_best.pth'}")
    print(f"  Training log:   {csv_path}")


if __name__ == "__main__":
    main()
