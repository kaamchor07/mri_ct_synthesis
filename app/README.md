---
title: MRI to Synthetic CT Synthesis
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
license: mit
---

# MRI → Synthetic CT — HuggingFace Spaces Deployment Guide

This document explains how to deploy the Gradio demo to HuggingFace Spaces.

## Prerequisites

1. A [HuggingFace account](https://huggingface.co/join) (free).
2. A trained model checkpoint: `checkpoints/pix2pix_best.pth`
3. Git and the `huggingface_hub` Python package:
   ```
   pip install huggingface_hub
   ```

---

## Step 1 — Create a new Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `mri-ct-synthesis` (or any name)
   - **SDK**: Gradio
   - **Visibility**: Public or Private
3. Click **Create Space**.

---

## Step 2 — Upload the required files

The Space only needs these files (not the full training stack):

```
app/
└── app.py            ← main Gradio app (this directory's app.py)
src/
├── model_unet.py     ← U-Net architecture (imported by app.py)
└── utils.py          ← normalization helpers (imported by app.py)
checkpoints/
└── pix2pix_best.pth  ← trained generator weights
requirements.txt      ← use requirements_deploy.txt contents
README.md             ← the YAML header at the top of this file
```

> **Tip**: Rename `requirements_deploy.txt` to `requirements.txt` before
> uploading to Spaces — Spaces reads `requirements.txt` by default.

---

## Step 3 — Upload via Git

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/mri-ct-synthesis
cd mri-ct-synthesis

# Copy required files (from your local project)
copy ..\app\app.py app.py
mkdir src
copy ..\src\model_unet.py src\model_unet.py
copy ..\src\utils.py src\utils.py
mkdir checkpoints
copy ..\checkpoints\pix2pix_best.pth checkpoints\pix2pix_best.pth
copy ..\requirements_deploy.txt requirements.txt
copy ..\app\README.md README.md

# Commit and push
git add .
git commit -m "Initial Spaces deployment"
git push
```

---

## Step 4 — Large file support (for checkpoint)

The checkpoint may be >100 MB. Use Git LFS:

```bash
# Install Git LFS (Windows)
winget install -e --id GitHub.GitLFS

# Initialize LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add checkpoints/pix2pix_best.pth
git commit -m "Add model checkpoint via LFS"
git push
```

---

## Step 5 — Configure the app file path

The YAML header at the top of this README sets `app_file: app.py`.
HuggingFace Spaces will automatically run:

```bash
python app.py
```

The `--checkpoint` defaults to `checkpoints/pix2pix_best.pth` which matches
the path used in Step 2.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: model_unet` | Ensure `src/` is in the Space repo |
| `FileNotFoundError: checkpoint` | Confirm `checkpoints/pix2pix_best.pth` is uploaded |
| Slow inference on Spaces CPU | Normal — CPU inference takes ~2–5 seconds per slice |
| Memory errors | Upgrade to a GPU Space (paid) or reduce model size |

---

## Notes

- Spaces free tier uses CPU — inference is slower but fully functional.
- The disclaimer ("not for clinical use") is displayed prominently in the UI.
- The app uses `parse_known_args()` so Gradio's injected arguments don't cause errors.
