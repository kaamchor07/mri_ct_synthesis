# data/README.md

# SynthRAD2023 Dataset — Download & Setup Instructions

This project uses the **SynthRAD2023** dataset: paired MRI + CT brain scans that are
already co-registered (aligned). Using pre-registered data means we can train a
supervised image-to-image translation model without needing to solve the registration
problem ourselves.

---

## Dataset Overview

| Property | Details |
|----------|---------|
| Dataset name | SynthRAD2023 Grand Challenge |
| Subsets | Brain (smaller, recommended to start) + Pelvis |
| Total brain cases | ~180 patients |
| Data format | NIfTI (.nii.gz) |
| Registration | Pre-registered (MRI and CT in the same coordinate space) |
| License | Research only — see the challenge website for terms |
| Publication | Thummerer et al. (2023), *Medical Physics* |

---

## Step 1 — Register for access

1. Go to: **https://synthrad2023.grand-challenge.org/**
2. Click **"Participate"** → Create a Grand Challenge account (free).
3. Accept the data usage agreement (research use only).
4. After approval (usually instant), you will receive a download link or access
   to their [Zenodo page](https://zenodo.org/records/7260428).

> ⚠️  **Important**: By downloading this dataset you agree to use it for
> non-commercial research only. Do NOT use it for clinical purposes.

---

## Step 2 — Download the brain subset

The brain data is approximately **4–6 GB** compressed.

### Option A: Download via browser
1. Log in to the Grand Challenge platform.
2. Navigate to the **Data** tab of the challenge.
3. Download `brain.zip` (or the equivalent archive).

### Option B: Download via command line (Windows PowerShell)

Once you have the direct download URL from the challenge platform:

```powershell
# Create the data directory
New-Item -ItemType Directory -Force -Path .\data\brain

# Download using curl (built into Windows 10/11)
curl.exe -L -o .\data\brain.zip "PASTE_YOUR_DOWNLOAD_URL_HERE"

# Extract the archive
Expand-Archive -Path .\data\brain.zip -DestinationPath .\data\brain -Force

# Remove the zip file to save space
Remove-Item .\data\brain.zip
```

### Option C: Download via Python (using the Grand Challenge API)

```python
# Install the grand-challenge client:
#   pip install gcapi
import gcapi
client = gcapi.Client(token="YOUR_API_TOKEN")
# Follow the gcapi docs at https://grand-challenge.org/documentation/gcapi/
```

---

## Step 3 — Verify the folder structure

After extraction, your `data/` directory should look like this:

```
data/
└── brain/
    ├── 1BA001/
    │   ├── mr.nii.gz      ← T1-weighted MRI volume
    │   └── ct.nii.gz      ← CT volume (co-registered to MRI)
    ├── 1BA002/
    │   ├── mr.nii.gz
    │   └── ct.nii.gz
    ├── 1BA003/
    │   └── ...
    └── ... (approximately 180 patient folders)
```

Each patient folder is named with a unique anonymized ID (e.g. `1BA001`).

---

## Step 4 — Quick verification

Run this Python snippet to confirm the data loaded correctly:

```python
import nibabel as nib
from pathlib import Path

# Load one patient
patient = list(Path('data/brain').iterdir())[0]
mri = nib.load(str(patient / 'mr.nii.gz'))
ct  = nib.load(str(patient / 'ct.nii.gz'))

print(f'Patient: {patient.name}')
print(f'MRI shape: {mri.shape}')  # Expected: (X, Y, ~200)
print(f'CT shape:  {ct.shape}')   # Should match MRI shape
print(f'MRI spacing: {mri.header.get_zooms()} mm')
print(f'CT spacing:  {ct.header.get_zooms()} mm')
```

You should see matching shapes like `(240, 240, 175)` and spacing close to `(1.0, 1.0, 1.0)`.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| Download link expired | Log back in to Grand Challenge and regenerate the link |
| `mr.nii.gz` not found | The file may be named `mri.nii.gz` or `t1.nii.gz` — `dataset.py` checks all names automatically |
| Shape mismatch between MRI and CT | Some patients may have slightly different shapes; the dataset loader skips these gracefully |
| `FileNotFoundError` when training | Double-check that your `--data_dir` points to `data/brain/`, not `data/` |
| Very slow training | Normal for first epoch (data is cached); use `--num_workers 0` on Windows if you get multiprocessing errors |

---

## Citation

If you use this dataset in any publication or portfolio project, please cite:

```
Thummerer, A., et al. (2023).
SynthRAD2023 Grand Challenge dataset.
Zenodo. https://doi.org/10.5281/zenodo.7260428

Thummerer, A., et al. (2023).
SynthRAD2023: A large multi-center dataset for synthesis of radiotherapy
planning CT from MRI.
Medical Physics. https://doi.org/10.1002/mp.16529
```
