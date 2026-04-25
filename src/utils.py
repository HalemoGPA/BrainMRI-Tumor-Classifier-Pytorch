from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

LABELS = ["No Tumor", "Pituitary", "Glioma", "Meningioma"]
NUM_DISPLAY_CLASSES = len(LABELS)


def is_likely_mri(pil_image: Image.Image) -> tuple[bool, dict]:
    """Heuristic check: brain MRIs are near-grayscale with dark backgrounds.

    Returns (is_mri_like, diagnostics).
    """
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return False, {"reason": "not RGB"}
    # Per-pixel saturation: how far the max channel is from the min channel.
    chroma = arr.max(axis=-1) - arr.min(axis=-1)
    mean_chroma = float(chroma.mean())
    # Background darkness: real MRIs have a lot of near-black pixels.
    luma = arr.mean(axis=-1)
    dark_fraction = float((luma < 25).mean())
    diagnostics = {
        "mean_chroma": mean_chroma,
        "dark_fraction": dark_fraction,
    }
    # Thresholds tuned empirically: real grayscale MRIs sit at chroma < 8,
    # color logos / photos easily exceed 25. Background should cover >5% of image.
    is_mri = mean_chroma < 15 and dark_fraction > 0.05
    return is_mri, diagnostics


def predict(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


def predict_with_probs(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits[:, :NUM_DISPLAY_CLASSES], dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    return top_idx, probs


def _pil_to_normalized_array(pil_image: Image.Image, size: int = 224) -> np.ndarray:
    resized = pil_image.convert("RGB").resize((size, size))
    return np.asarray(resized, dtype=np.float32) / 255.0


def compute_gradcam(model, image_tensor, target_layer, predicted_class: int, original_pil: Image.Image) -> Image.Image:
    cam = GradCAM(model=model, target_layers=[target_layer])
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=[ClassifierOutputTarget(predicted_class)],
    )[0]
    rgb_image = _pil_to_normalized_array(original_pil)
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay)


def make_pdf_report(
    *,
    original_image: Image.Image,
    heatmap_image: Image.Image,
    predicted_label: str,
    probabilities: np.ndarray,
    model_version: str = "model_38",
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 18 * mm

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, height - margin, "Brain MRI Tumor Classification Report")

    c.setFont("Helvetica", 10)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    c.drawString(margin, height - margin - 6 * mm, f"Generated: {timestamp}    Model: {model_version}")
    c.line(margin, height - margin - 8 * mm, width - margin, height - margin - 8 * mm)

    img_size = 75 * mm
    img_y = height - margin - 12 * mm - img_size

    def _draw(img: Image.Image, x: float, label: str) -> None:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, img_y + img_size + 3 * mm, label)
        reader = ImageReader(img.convert("RGB"))
        c.drawImage(reader, x, img_y, width=img_size, height=img_size, preserveAspectRatio=True, anchor="c")

    _draw(original_image, margin, "Uploaded MRI")
    _draw(heatmap_image, margin + img_size + 8 * mm, "Grad-CAM heatmap")

    table_y = img_y - 14 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, table_y, f"Prediction: {predicted_label}")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, table_y - 10 * mm, "Class")
    c.drawString(margin + 70 * mm, table_y - 10 * mm, "Probability")
    c.setFont("Helvetica", 11)
    for i, label in enumerate(LABELS):
        row_y = table_y - (14 + 6 * i) * mm
        c.drawString(margin, row_y, label)
        c.drawString(margin + 70 * mm, row_y, f"{probabilities[i] * 100:.2f}%")

    disclaimer_y = margin + 22 * mm
    c.setFont("Helvetica-Oblique", 9)
    c.setFillGray(0.35)
    c.drawString(margin, disclaimer_y, "Disclaimer:")
    c.setFont("Helvetica", 9)
    c.drawString(
        margin,
        disclaimer_y - 5 * mm,
        "For educational and research use only. Not a medical device.",
    )
    c.drawString(
        margin,
        disclaimer_y - 10 * mm,
        "Do not use these predictions for diagnosis or treatment decisions.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()
