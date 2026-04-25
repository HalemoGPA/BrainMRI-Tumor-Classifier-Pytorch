from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib.colors import Color, HexColor
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


# Brand palette mirrors app.py CSS variables.
_ACCENT_1 = HexColor("#7c3aed")  # purple
_ACCENT_2 = HexColor("#06b6d4")  # cyan
_INK = HexColor("#0f172a")
_INK_SOFT = HexColor("#475569")
_INK_FAINT = HexColor("#94a3b8")
_BG_SOFT = HexColor("#f1f5f9")
_BG_CARD = HexColor("#faf5ff")
_BORDER = HexColor("#e2e8f0")
_GOOD = HexColor("#10b981")
_WARN = HexColor("#f59e0b")
_BAD = HexColor("#ef4444")


def _confidence_color(p: float) -> Color:
    if p >= 0.85:
        return _GOOD
    if p >= 0.6:
        return _WARN
    return _BAD


def _confidence_label(p: float) -> str:
    if p >= 0.85:
        return "High confidence"
    if p >= 0.6:
        return "Medium confidence"
    return "Low confidence"


def _draw_gradient_band(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                        c1: Color, c2: Color, steps: int = 60) -> None:
    """Horizontal gradient as thin vertical strips."""
    strip = w / steps
    for i in range(steps):
        t = i / (steps - 1)
        r = c1.red + (c2.red - c1.red) * t
        g = c1.green + (c2.green - c1.green) * t
        b = c1.blue + (c2.blue - c1.blue) * t
        c.setFillColorRGB(r, g, b)
        c.rect(x + i * strip, y, strip + 0.5, h, stroke=0, fill=1)


def _draw_image_card(c: canvas.Canvas, img: Image.Image, x: float, y: float,
                     w: float, h: float, label: str) -> None:
    # Card label (above image)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(_INK_SOFT)
    c.drawString(x, y + h + 3 * mm, label.upper())

    # Card border
    c.setStrokeColor(_BORDER)
    c.setLineWidth(0.7)
    c.roundRect(x - 1.5, y - 1.5, w + 3, h + 3, 3 * mm, stroke=1, fill=0)

    # Image
    reader = ImageReader(img.convert("RGB"))
    c.drawImage(reader, x, y, width=w, height=h, preserveAspectRatio=True, anchor="c", mask="auto")


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

    top_idx = int(np.argmax(probabilities))
    top_prob = float(probabilities[top_idx])

    # ---- Header band -------------------------------------------------------
    band_h = 26 * mm
    band_y = height - band_h
    _draw_gradient_band(c, 0, band_y, width, band_h, _ACCENT_1, _ACCENT_2)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, band_y + band_h - 12 * mm, "Brain MRI Tumor Classification")
    c.setFont("Helvetica", 10)
    c.drawString(margin, band_y + band_h - 18 * mm, "Grad-CAM-explained PyTorch CNN report")

    # Right-aligned meta
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c.setFont("Helvetica", 9)
    meta_x = width - margin
    c.drawRightString(meta_x, band_y + band_h - 12 * mm, f"Generated  {timestamp}")
    c.drawRightString(meta_x, band_y + band_h - 17 * mm, f"Model  {model_version}")

    # ---- Image cards -------------------------------------------------------
    img_y = band_y - 8 * mm - 70 * mm
    img_w = (width - 2 * margin - 8 * mm) / 2
    img_h = 70 * mm

    _draw_image_card(c, original_image, margin, img_y, img_w, img_h, "Uploaded MRI")
    _draw_image_card(c, heatmap_image, margin + img_w + 8 * mm, img_y, img_w, img_h,
                     "Grad-CAM heatmap")

    # ---- Prediction card ---------------------------------------------------
    card_y = img_y - 12 * mm - 38 * mm
    card_h = 38 * mm
    card_w = width - 2 * margin

    # Card background
    c.setFillColor(_BG_CARD)
    c.setStrokeColor(_ACCENT_1)
    c.setLineWidth(0.7)
    c.roundRect(margin, card_y, card_w, card_h, 4 * mm, stroke=1, fill=1)

    # Left accent stripe
    c.setFillColor(_ACCENT_1)
    c.rect(margin, card_y, 2 * mm, card_h, stroke=0, fill=1)

    inner_x = margin + 8 * mm
    inner_top = card_y + card_h - 8 * mm

    # "PREDICTION" pill
    pill_w, pill_h = 26 * mm, 5 * mm
    _draw_gradient_band(c, inner_x, inner_top - pill_h + 1 * mm, pill_w, pill_h,
                        _ACCENT_1, _ACCENT_2, steps=30)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(inner_x + pill_w / 2, inner_top - pill_h + 2.6 * mm, "PREDICTION")

    # Label
    c.setFillColor(_INK)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(inner_x, inner_top - 16 * mm, predicted_label)

    # Confidence (right side)
    conf_color = _confidence_color(top_prob)
    c.setFillColor(_INK_SOFT)
    c.setFont("Helvetica", 9)
    c.drawRightString(margin + card_w - 8 * mm, inner_top - 4 * mm,
                      _confidence_label(top_prob).upper())
    c.setFillColor(conf_color)
    c.setFont("Helvetica-Bold", 26)
    c.drawRightString(margin + card_w - 8 * mm, inner_top - 14 * mm,
                      f"{top_prob * 100:.1f}%")

    # Class description
    desc_map = {
        "No Tumor": "Healthy brain tissue. No tumor detected in the scan.",
        "Pituitary": "Tumor in the pituitary gland (base of the brain). Often benign.",
        "Glioma": "Tumor arising from glial cells. Can be aggressive; needs urgent review.",
        "Meningioma": "Tumor of the meninges (brain's protective layers). Usually slow-growing.",
    }
    c.setFillColor(_INK_SOFT)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(inner_x, card_y + 6 * mm, desc_map.get(predicted_label, ""))

    # ---- Per-class probability bars ----------------------------------------
    bars_y = card_y - 14 * mm
    c.setFillColor(_INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, bars_y, "Per-class probability")

    bar_top = bars_y - 6 * mm
    row_h = 9 * mm
    name_w = 30 * mm
    pct_w = 18 * mm
    bar_x = margin + name_w
    bar_w = card_w - name_w - pct_w
    bar_h = 5 * mm

    for i, label in enumerate(LABELS):
        row_y = bar_top - (i + 1) * row_h + (row_h - bar_h) / 2
        text_y = row_y + (bar_h - 8) / 2 + 0.5
        prob = float(probabilities[i])

        # name
        c.setFillColor(_INK)
        c.setFont("Helvetica" if i != top_idx else "Helvetica-Bold", 10)
        c.drawString(margin, text_y, label)

        # bg track
        c.setFillColor(_BG_SOFT)
        c.roundRect(bar_x, row_y, bar_w, bar_h, bar_h / 2, stroke=0, fill=1)

        # filled portion
        fill_w = max(prob * bar_w, 0.5)
        if i == top_idx:
            _draw_gradient_band(c, bar_x, row_y, fill_w, bar_h, _ACCENT_1, _ACCENT_2,
                                steps=max(int(fill_w / 2), 4))
        else:
            c.setFillColor(_INK_FAINT)
            c.roundRect(bar_x, row_y, fill_w, bar_h, bar_h / 2, stroke=0, fill=1)

        # pct
        c.setFillColor(_INK)
        c.setFont("Helvetica" if i != top_idx else "Helvetica-Bold", 10)
        c.drawRightString(margin + card_w, text_y, f"{prob * 100:.2f}%")

    # ---- Footer disclaimer -------------------------------------------------
    footer_h = 18 * mm
    c.setFillColor(_BG_SOFT)
    c.rect(0, 0, width, footer_h, stroke=0, fill=1)
    c.setStrokeColor(_BORDER)
    c.setLineWidth(0.5)
    c.line(0, footer_h, width, footer_h)

    c.setFillColor(_WARN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, footer_h - 7 * mm, "DISCLAIMER")
    c.setFillColor(_INK_SOFT)
    c.setFont("Helvetica", 9)
    c.drawString(
        margin + 22 * mm,
        footer_h - 7 * mm,
        "For educational and research use only. Not a medical device.",
    )
    c.setFillColor(_INK_FAINT)
    c.setFont("Helvetica", 8)
    c.drawString(
        margin,
        footer_h - 12 * mm,
        "Do not use these predictions for diagnosis or treatment decisions. "
        "Consult a qualified clinician for any medical concerns.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()
