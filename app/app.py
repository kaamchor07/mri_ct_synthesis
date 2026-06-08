"""
app/app.py
==========
Gradio demo for MRI → synthetic CT image translation.

This app allows anyone to upload a 2D MRI slice (PNG or JPG) and instantly
see the generated synthetic CT (sCT) produced by the trained pix2pix model.

⚠️  DISCLAIMER: This is a research demonstration only.
    It is NOT validated for clinical use. Do not use it for
    medical diagnosis or treatment planning.

How to run locally
------------------
    python app/app.py --checkpoint checkpoints/pix2pix_best.pth

How to deploy on HuggingFace Spaces
------------------------------------
    1. Push this repo to a HuggingFace Space (see app/README.md).
    2. The Space will auto-install requirements_deploy.txt and launch this file.

Input:  PNG or JPG of a brain MRI slice (grayscale or RGB)
Output: Grayscale synthetic CT slice rendered as a PIL image
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

# When running as a standalone Spaces app, src/ may not be in the path.
# Add it explicitly so we can import the model and utils.
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model_unet import UNet   # Generator architecture (same for both models)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SIZE: int = 256          # Must match training resolution
DEFAULT_CHECKPOINT: str = str(ROOT_DIR / "checkpoints" / "pix2pix_best.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing (supports --checkpoint flag)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Gradio app.

    Returns:
        argparse.Namespace with 'checkpoint' and 'port' fields.
    """
    parser = argparse.ArgumentParser(
        description="Gradio demo: MRI → Synthetic CT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to the trained generator checkpoint (.pth).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Local port to serve the Gradio app.",
    )
    # Gradio injects extra args in Spaces — ignore unknown args gracefully
    args, _ = parser.parse_known_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

# Global model cache — load once, reuse for all inference calls
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load the generator from a checkpoint file (cached after first call).

    Args:
        checkpoint_path: Path to the .pth generator weights file.

    Returns:
        UNet model in eval mode on the best available device.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    global _model, _device

    if _model is not None:
        return _model  # Return cached model

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train the model first with: python src/train_pix2pix.py"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from %s onto %s", path, _device)

    model = UNet(in_channels=1, out_channels=1)
    # Handle both plain state_dict and wrapped checkpoints
    ckpt = torch.load(path, map_location=_device)
    if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
        model.load_state_dict(ckpt["generator_state_dict"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(_device).eval()
    _model = model
    logger.info("Model ready.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pre/post processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert an uploaded PIL image into a normalized MRI tensor.

    Steps:
      1. Convert to grayscale (in case the user uploads an RGB image).
      2. Resize to 256×256.
      3. Scale to [0, 1].
      4. Add batch and channel dimensions → (1, 1, 256, 256).

    Args:
        pil_image: PIL Image uploaded by the user (any mode).

    Returns:
        Float32 tensor of shape (1, 1, 256, 256) on the model's device.
    """
    # Convert to grayscale (L mode)
    img = pil_image.convert("L")

    # Resize to training resolution
    img = img.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Percentile clip to remove extreme outliers (mimics training normalization)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)
    else:
        arr = np.zeros_like(arr)

    # (H, W) → (1, 1, H, W): batch and channel dimensions
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor.to(_device)


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a model output tensor to a displayable grayscale PIL image.

    Args:
        tensor: Model output of shape (1, 1, H, W), values in [−1, 1].

    Returns:
        Grayscale PIL Image scaled to [0, 255].
    """
    arr = tensor.squeeze().detach().cpu().numpy()  # (H, W), values in [−1, 1]

    # Map [−1, 1] → [0, 255] for display
    arr = (arr + 1.0) / 2.0 * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")


# ─────────────────────────────────────────────────────────────────────────────
# Inference function (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_ct(input_image: Image.Image, checkpoint_path: str) -> Image.Image:
    """
    Run MRI → sCT inference on an uploaded image.

    This is the function called by Gradio on every user submission.

    Args:
        input_image:     PIL Image uploaded by the user.
        checkpoint_path: Path to the checkpoint (set at app launch).

    Returns:
        Grayscale PIL Image of the synthetic CT slice.
    """
    if input_image is None:
        raise gr.Error("Please upload an MRI slice image first.")

    model = load_model(checkpoint_path)

    with torch.no_grad():
        mri_tensor = preprocess_image(input_image)
        pred_ct_tensor = model(mri_tensor)

    return postprocess_tensor(pred_ct_tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_interface(checkpoint_path: str) -> gr.Blocks:
    """
    Build and return the Gradio Blocks interface.

    Args:
        checkpoint_path: Path to the generator checkpoint (injected via closure).

    Returns:
        Configured gr.Blocks interface.
    """
    # Pre-load model when the app starts (avoids first-request latency)
    try:
        load_model(checkpoint_path)
        model_status = f"✅ Model loaded from `{checkpoint_path}`"
    except FileNotFoundError as e:
        model_status = f"⚠️  {e}"

    with gr.Blocks(
        title="MRI → Synthetic CT Demo",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 16px;
            border-radius: 4px;
            font-size: 0.9em;
            margin-bottom: 12px;
        }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🧠 MRI → Synthetic CT Synthesis
            Upload a **2D brain MRI slice** (PNG or JPG) and the model will generate
            a **synthetic CT** slice using a trained pix2pix conditional GAN.

            This demo is part of a portfolio project. [View source on GitHub](#)
            """
        )

        gr.HTML(
            '<div class="disclaimer">'
            "⚠️ <strong>Research demo only — NOT for clinical use.</strong> "
            "This model has not been validated for medical diagnosis or treatment planning. "
            "Synthetic CT images may contain hallucinations or structural errors. "
            "Always consult a licensed radiologist for medical decisions."
            "</div>"
        )

        gr.Markdown(f"*{model_status}*")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="🔬 MRI Input (upload PNG or JPG)",
                    type="pil",
                    image_mode="L",
                )
                submit_btn = gr.Button("Generate Synthetic CT ⚡", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="🩻 Synthetic CT Output",
                    type="pil",
                    image_mode="L",
                )

        gr.Markdown(
            """
            ### How to use
            1. Upload a grayscale PNG/JPG of a brain MRI axial slice.
            2. Click **Generate Synthetic CT**.
            3. The right panel will show the predicted CT slice.

            ### Notes
            - Input is automatically resized to 256×256 pixels.
            - For best results, use axial slices from brain MRI scans similar
              to the **SynthRAD2023** training distribution.
            - The model was trained on paired MRI + CT data. It cannot generalize
              to other body regions or modalities without retraining.
            """
        )

        # Wire up the button
        submit_btn.click(
            fn=lambda img: synthesize_ct(img, checkpoint_path),
            inputs=[input_img],
            outputs=[output_img],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    demo = build_interface(checkpoint_path=args.checkpoint)
    demo.launch(server_port=args.port, share=False)
