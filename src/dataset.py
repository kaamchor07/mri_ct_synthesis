"""
src/dataset.py
==============
PyTorch Dataset and DataLoader factory for paired MRI–CT NIfTI volumes.

High-level flow
---------------
1.  Scan the data directory for patient folders, each containing an
    ``mri.nii.gz`` and a ``ct.nii.gz`` file.
2.  Split patients into train / val / test sets (70 / 15 / 15 %).
3.  For each patient, extract 2D axial slices from the 3-D volumes.
4.  Skip near-empty slices (>95 % background pixels).
5.  Normalize MRI → [0, 1]  and  CT → [−1, 1].
6.  Resize every slice to TARGET_SIZE × TARGET_SIZE (256 × 256 by default).
7.  Return (mri_tensor, ct_tensor) pairs with shape (1, H, W).

Expected data directory layout (SynthRAD2023 brain subset)
-----------------------------------------------------------
data/
└── brain/
    ├── 1BA001/
    │   ├── mr.nii.gz   (or mri.nii.gz)
    │   └── ct.nii.gz
    ├── 1BA002/
    │   └── ...
    └── ...

The exact MRI filename (``mr.nii.gz`` vs ``mri.nii.gz``) is detected
automatically — see MRI_FILENAMES below.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils import normalize_ct, normalize_mri

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE: int = 256          # Output spatial resolution (H and W)
BACKGROUND_THRESHOLD: float = 0.95  # Skip slice if this fraction is background
BACKGROUND_VALUE: float = 0.02  # Pixels below this (post-norm) are "background"

# SynthRAD2023 may name the MRI file differently — we check both
MRI_FILENAMES: Tuple[str, ...] = ("mr.nii.gz", "mri.nii.gz", "t1.nii.gz")
CT_FILENAME: str = "ct.nii.gz"

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# TEST_RATIO is implicitly 1 − TRAIN_RATIO − VAL_RATIO = 0.15

RANDOM_SEED: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_patient_dirs(data_root: Path) -> List[Path]:
    """
    Recursively discover all patient directories that contain a CT file.

    A patient directory is any leaf folder that contains ``ct.nii.gz``.

    Args:
        data_root: Root of the dataset (e.g. ``data/brain/``).

    Returns:
        Sorted list of Path objects, one per patient.
    """
    patient_dirs = sorted([
        p.parent
        for p in data_root.rglob(CT_FILENAME)
    ])
    logger.info("Found %d patient directories under %s", len(patient_dirs), data_root)
    return patient_dirs


def _find_mri_file(patient_dir: Path) -> Optional[Path]:
    """
    Find the MRI NIfTI file inside a patient directory.

    Checks MRI_FILENAMES in order and returns the first match.

    Args:
        patient_dir: Path to a single patient folder.

    Returns:
        Path to the MRI file, or None if not found.
    """
    for name in MRI_FILENAMES:
        candidate = patient_dir / name
        if candidate.exists():
            return candidate
    return None


def _resize_slice(arr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Resize a 2D numpy array to (size × size) using bilinear interpolation.

    Uses PIL for bilinear resize, which is higher quality than nearest-
    neighbor and does not require scipy.

    Args:
        arr:  2D float32 array (H, W).
        size: Target edge length in pixels.

    Returns:
        Float32 numpy array of shape (size, size).
    """
    # PIL expects uint8 or float images; we use mode="F" for float32
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img_resized = img.resize((size, size), resample=Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def _is_near_empty(slice_arr: np.ndarray, threshold: float = BACKGROUND_THRESHOLD) -> bool:
    """
    Return True if a 2D slice is mostly background and should be skipped.

    "Background" is defined as pixels with normalized value below
    BACKGROUND_VALUE. This filters out the many all-air slices at the
    top and bottom of a head scan.

    Args:
        slice_arr:  2D normalized array (values in [0, 1] for MRI).
        threshold:  Fraction of pixels that must be background to skip.

    Returns:
        True if the slice should be skipped.
    """
    background_fraction = np.mean(slice_arr < BACKGROUND_VALUE)
    return bool(background_fraction > threshold)


def _extract_slices_from_patient(
    patient_dir: Path,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one patient's MRI+CT volumes and extract normalized 2D axial slices.

    Steps:
      1. Load .nii.gz volumes with nibabel.
      2. Normalize MRI (percentile clip → [0,1]) and CT (HU clip → [−1,1]).
      3. Extract axial slices (last axis = z).
      4. Skip near-empty slices.
      5. Resize each slice to TARGET_SIZE × TARGET_SIZE.

    Args:
        patient_dir: Path to a patient folder with mr.nii.gz + ct.nii.gz.

    Returns:
        List of (mri_slice, ct_slice) tuples, each of shape (TARGET_SIZE, TARGET_SIZE).
        Returns an empty list if MRI or CT file is missing.
    """
    mri_path = _find_mri_file(patient_dir)
    ct_path = patient_dir / CT_FILENAME

    if mri_path is None:
        logger.warning("No MRI file found in %s — skipping patient.", patient_dir)
        return []

    if not ct_path.exists():
        logger.warning("No CT file found in %s — skipping patient.", patient_dir)
        return []

    # Load volumes (nibabel returns (X, Y, Z) arrays; z-axis = axial)
    mri_vol = nib.load(str(mri_path)).get_fdata(dtype=np.float32)
    ct_vol = nib.load(str(ct_path)).get_fdata(dtype=np.float32)

    # Normalize full volumes before slicing (percentiles are volume-wide)
    mri_norm = normalize_mri(mri_vol)
    ct_norm = normalize_ct(ct_vol)

    n_slices = mri_vol.shape[2]
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for z in range(n_slices):
        mri_slice = mri_norm[:, :, z]
        ct_slice = ct_norm[:, :, z]

        # Skip slices that are almost entirely background (air)
        if _is_near_empty(mri_slice):
            continue

        # Resize to TARGET_SIZE × TARGET_SIZE
        mri_slice = _resize_slice(mri_slice)
        ct_slice = _resize_slice(ct_slice)

        pairs.append((mri_slice, ct_slice))

    logger.debug(
        "Patient %s: extracted %d/%d non-empty slices.",
        patient_dir.name, len(pairs), n_slices,
    )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class MRICTDataset(Dataset):
    """
    PyTorch Dataset for paired MRI–CT 2D axial slices.

    Each item is a tuple ``(mri_tensor, ct_tensor)`` where:
      - ``mri_tensor`` has shape ``(1, 256, 256)`` and values in ``[0, 1]``.
      - ``ct_tensor`` has shape ``(1, 256, 256)`` and values in ``[−1, 1]``.

    The dataset is split by **patient** (not by slice) to prevent data
    leakage between train and test sets. If patients from the same subject
    appeared in both train and test, the model could memorize anatomy
    rather than learning to synthesize it.

    Args:
        data_root:   Root directory of the dataset (e.g. ``data/brain/``).
        split:       One of ``"train"``, ``"val"``, or ``"test"``.
        seed:        Random seed for reproducible patient splits.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}, \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_root = Path(data_root)
        self.split = split

        # Discover and shuffle patients deterministically
        all_patients = _find_patient_dirs(self.data_root)
        if len(all_patients) == 0:
            raise FileNotFoundError(
                f"No patient directories with '{CT_FILENAME}' found under {self.data_root}. "
                "Please follow data/README.md to download the SynthRAD2023 dataset."
            )

        rng = random.Random(seed)
        patients = list(all_patients)
        rng.shuffle(patients)

        # Compute split indices
        n = len(patients)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # The test set gets whatever is left over

        if split == "train":
            selected_patients = patients[:n_train]
        elif split == "val":
            selected_patients = patients[n_train : n_train + n_val]
        else:  # test
            selected_patients = patients[n_train + n_val :]

        logger.info(
            "Split '%s': %d patients selected (total %d).",
            split, len(selected_patients), n,
        )

        # Load all slices for selected patients
        self.slices: List[Tuple[np.ndarray, np.ndarray]] = []
        for patient_dir in selected_patients:
            self.slices.extend(_extract_slices_from_patient(patient_dir))

        logger.info(
            "Dataset '%s' ready: %d slices loaded.", split, len(self.slices)
        )

    def __len__(self) -> int:
        """Return the total number of 2D slices in this split."""
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieve a single (MRI, CT) slice pair as tensors.

        Args:
            idx: Integer index into the slice list.

        Returns:
            Tuple of (mri_tensor, ct_tensor), each with shape (1, 256, 256).
        """
        mri_slice, ct_slice = self.slices[idx]

        # Add channel dimension: (H, W) → (1, H, W)
        mri_tensor = torch.from_numpy(mri_slice).unsqueeze(0)  # (1, 256, 256)
        ct_tensor = torch.from_numpy(ct_slice).unsqueeze(0)    # (1, 256, 256)

        return mri_tensor, ct_tensor


# We need Union for Python 3.9 compatibility
from typing import Union  # noqa: E402 — must be after dataset class to avoid circular ref


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    data_root: Union[str, Path],
    split: str,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = RANDOM_SEED,
) -> DataLoader:
    """
    Create a DataLoader for a given dataset split.

    Args:
        data_root:   Root of the dataset (e.g. ``data/brain/``).
        split:       ``"train"``, ``"val"``, or ``"test"``.
        batch_size:  Number of slice pairs per batch.
        num_workers: CPU workers for parallel data loading.
                     Use 0 on Windows if you get multiprocessing errors.
        seed:        Random seed for patient split reproducibility.

    Returns:
        A configured torch.utils.data.DataLoader.
    """
    dataset = MRICTDataset(data_root=data_root, split=split, seed=seed)
    shuffle = (split == "train")  # Only shuffle training data

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,   # Faster GPU transfers when using CUDA
        drop_last=(split == "train"),  # Avoid partial batches during training
    )
    logger.info(
        "DataLoader '%s': %d batches of size %d.",
        split, len(loader), batch_size,
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (used by Gradio app)
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_pair(
    data_root: Union[str, Path],
    patient_idx: int = 0,
    slice_idx: int = 50,
) -> Tuple[Tensor, Tensor]:
    """
    Load a single (MRI, CT) pair without creating a full DataLoader.

    Useful for quick inference in the Gradio app or exploratory notebooks.
    No shuffling is applied — the same indices always return the same pair.

    Args:
        data_root:   Root of the dataset (e.g. ``data/brain/``).
        patient_idx: Which patient to load (0-indexed, sorted alphabetically).
        slice_idx:   Which axial slice index within the patient volume.

    Returns:
        Tuple of (mri_tensor, ct_tensor), each with shape (1, 256, 256).
        Both tensors are on CPU with dtype float32.

    Raises:
        FileNotFoundError: If no patient directories are found.
        IndexError:        If patient_idx or slice_idx are out of range.
    """
    data_root = Path(data_root)
    patients = _find_patient_dirs(data_root)
    if not patients:
        raise FileNotFoundError(f"No patient dirs found under {data_root}")

    patient_dir = patients[patient_idx]
    slices = _extract_slices_from_patient(patient_dir)
    if not slices:
        raise ValueError(f"No usable slices found for patient {patient_dir.name}")

    # Clamp slice_idx to valid range
    slice_idx = min(slice_idx, len(slices) - 1)
    mri_slice, ct_slice = slices[slice_idx]

    mri_tensor = torch.from_numpy(mri_slice).unsqueeze(0)
    ct_tensor = torch.from_numpy(ct_slice).unsqueeze(0)
    return mri_tensor, ct_tensor
