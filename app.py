from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import get_gradcam_target_layer, load_model
from src.utils import (
    LABELS,
    compute_gradcam,
    make_pdf_report,
    predict_with_probs,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_38"
SAMPLE_DIR = ROOT / "sample"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = "model_38"
GITHUB_URL = "https://github.com/HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch"
DATASET_URL = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"

CLASS_DESCRIPTIONS = {
    "No Tumor": "Healthy brain tissue. No tumor detected in the scan.",
    "Pituitary": "Tumor in the pituitary gland (base of the brain). Often benign.",
    "Glioma": "Tumor arising from glial cells. Can be aggressive; needs urgent review.",
    "Meningioma": "Tumor of the meninges (brain's protective layers). Usually slow-growing.",
}

st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": GITHUB_URL,
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": "Brain MRI Tumor Classifier — PyTorch CNN with Grad-CAM explainability.",
    },
)

CSS = """
<style>
:root {
  --accent: #7c3aed;
  --accent-2: #06b6d4;
  --good: #10b981;
  --warn: #f59e0b;
  --bad: #ef4444;
}
.block-container { padding-top: 1.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
  background: linear-gradient(120deg, rgba(124,58,237,0.18), rgba(6,182,212,0.12));
  border: 1px solid rgba(124,58,237,0.35);
  border-radius: 16px;
  padding: 22px 28px;
  margin-bottom: 18px;
}
.hero h1 {
  font-size: 2.2rem;
  margin: 0 0 4px 0;
  background: linear-gradient(90deg, #c4b5fd, #67e8f9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero p { margin: 0; color: #b8c0d6; }

.disclaimer-banner {
  background: rgba(245, 158, 11, 0.08);
  border-left: 3px solid var(--warn);
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 0.86rem;
  color: #fcd34d;
  margin-bottom: 18px;
}

.section-title {
  display: flex; align-items: center; gap: 10px;
  font-size: 1.1rem; font-weight: 600; color: #e6e6e6;
  margin: 18px 0 10px 0;
}
.section-title .step {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: white; border-radius: 50%; font-size: 0.82rem; font-weight: 700;
}

.pred-card {
  background: rgba(124, 58, 237, 0.08);
  border: 1px solid rgba(124, 58, 237, 0.35);
  border-radius: 14px;
  padding: 18px 22px;
  margin-bottom: 14px;
}
.pred-card .label-tag {
  display: inline-block;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: white; padding: 4px 12px; border-radius: 999px;
  font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 8px;
}
.pred-card h2 { margin: 4px 0 10px 0; font-size: 2rem; }
.pred-card .conf { font-size: 0.92rem; color: #b8c0d6; margin-bottom: 6px; }
.pred-card .conf-val { font-size: 2.4rem; font-weight: 700; line-height: 1; }
.pred-card .desc { color: #c8cfde; font-size: 0.9rem; margin-top: 12px; line-height: 1.5; }

.prob-row {
  display: flex; align-items: center; gap: 12px; margin: 6px 0;
  font-size: 0.92rem;
}
.prob-row .name { width: 110px; color: #d6dbeb; }
.prob-row .bar {
  flex: 1; background: rgba(255,255,255,0.06);
  border-radius: 999px; height: 10px; overflow: hidden;
}
.prob-row .bar > div {
  height: 100%; border-radius: 999px;
  transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.prob-row .pct { width: 56px; text-align: right; color: #b8c0d6; font-variant-numeric: tabular-nums; }

div[data-testid="stImage"] img {
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

.sample-card { text-align: center; }
.sample-card img { border-radius: 10px; }

div[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(124, 58, 237, 0.4);
  border-radius: 12px;
  background: rgba(124, 58, 237, 0.04);
  transition: border-color 0.2s, background 0.2s;
}
div[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent);
  background: rgba(124, 58, 237, 0.08);
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border: none;
}
div.stDownloadButton > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border: none; color: white; font-weight: 600;
}

footer { visibility: hidden; }
</style>
"""

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model(str(MODEL_PATH), DEVICE)


@st.cache_data(show_spinner=False)
def list_sample_images() -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out.append((path.name, path.read_bytes()))
    return out


def preprocess(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image).unsqueeze(0)


def confidence_color(p: float) -> str:
    if p >= 0.85:
        return "var(--good)"
    if p >= 0.6:
        return "var(--warn)"
    return "var(--bad)"


def confidence_label(p: float) -> str:
    if p >= 0.85:
        return "High confidence"
    if p >= 0.6:
        return "Medium confidence"
    return "Low confidence"


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 Brain MRI Classifier")
        st.caption("PyTorch CNN with Grad-CAM explainability.")

        st.markdown("---")
        st.markdown("#### 🏗️ Model")
        st.markdown(
            f"`{MODEL_VERSION}` · 4 conv blocks → FC(512) → FC(num_classes)\n\n"
            f"**Input:** 224×224 RGB · **Device:** `{DEVICE}`"
        )

        st.markdown("#### 🏷️ Classes")
        for i, label in enumerate(LABELS):
            st.markdown(f"`{i}` · {label}")

        st.markdown("#### 📚 Resources")
        st.markdown(
            f"- [Dataset (Kaggle)]({DATASET_URL})\n"
            f"- [GitHub repository]({GITHUB_URL})"
        )

        st.markdown("---")
        if st.button("🔄 Start over", use_container_width=True):
            for k in ("mri_bytes", "mri_source"):
                st.session_state.pop(k, None)
            st.rerun()


def render_step_title(num: int, title: str) -> None:
    st.markdown(
        f'<div class="section-title"><span class="step">{num}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def render_input_section() -> Image.Image | None:
    render_step_title(1, "Choose an MRI image")

    col_upload, col_samples = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown("**Upload your scan**")
        uploaded = st.file_uploader(
            "Upload a brain MRI",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            accept_multiple_files=False,
        )
        if uploaded is not None:
            st.session_state["mri_bytes"] = uploaded.getvalue()
            st.session_state["mri_source"] = uploaded.name

    with col_samples:
        st.markdown("**Or try a sample**")
        samples = list_sample_images()
        sample_cols = st.columns(len(samples))
        for idx, (col, (name, data)) in enumerate(zip(sample_cols, samples), start=1):
            with col:
                st.image(data, use_container_width=True)
                if st.button(
                    f"▶ Sample {idx}",
                    key=f"use_{name}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state["mri_bytes"] = data
                    st.session_state["mri_source"] = f"Sample {idx}"

    data = st.session_state.get("mri_bytes")
    if not data:
        return None
    return Image.open(BytesIO(data)).convert("RGB")


def render_probability_bars(probs, top_idx: int) -> None:
    rows = []
    for i, label in enumerate(LABELS):
        pct = float(probs[i]) * 100
        if i == top_idx:
            color = "linear-gradient(90deg, var(--accent), var(--accent-2))"
        else:
            color = "rgba(255,255,255,0.18)"
        width = max(pct, 0.5)
        rows.append(
            f'<div class="prob-row">'
            f'<div class="name">{label}</div>'
            f'<div class="bar"><div style="width:{width}%; background:{color};"></div></div>'
            f'<div class="pct">{pct:.1f}%</div>'
            f"</div>"
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)


def render_results(image: Image.Image) -> None:
    with st.spinner("Analyzing scan…"):
        model = get_model()
        tensor = preprocess(image).to(DEVICE)
        top_idx, probs = predict_with_probs(model, tensor, DEVICE)
        target_layer = get_gradcam_target_layer(model)
        heatmap = compute_gradcam(model, tensor, target_layer, top_idx, image)

    label = LABELS[top_idx]
    confidence = float(probs[top_idx])
    conf_pct = confidence * 100

    render_step_title(2, "Result")

    col_pred, col_imgs = st.columns([1, 1.2], gap="large")

    with col_pred:
        st.markdown(
            f"""
            <div class="pred-card">
              <span class="label-tag">Prediction</span>
              <h2>{label}</h2>
              <div class="conf">{confidence_label(confidence)}</div>
              <div class="conf-val" style="color:{confidence_color(confidence)};">{conf_pct:.1f}%</div>
              <div class="desc">{CLASS_DESCRIPTIONS[label]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Per-class probability**")
        render_probability_bars(probs, top_idx)

    with col_imgs:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(
                image,
                caption=f"Uploaded · {st.session_state.get('mri_source', '')}",
                use_container_width=True,
            )
        with img_col2:
            st.image(
                heatmap,
                caption="Grad-CAM · model focus",
                use_container_width=True,
            )

    render_step_title(3, "Download report")
    pdf_bytes = make_pdf_report(
        original_image=image,
        heatmap_image=heatmap,
        predicted_label=label,
        probabilities=probs,
        model_version=MODEL_VERSION,
    )
    cdl, cdr = st.columns([1, 3])
    with cdl:
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=f"mri_report_{label.replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with cdr:
        st.caption(
            "The PDF includes the uploaded image, Grad-CAM heatmap, prediction, "
            "per-class probabilities, timestamp, and the disclaimer."
        )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(
        """
        <div class="hero">
          <h1>🧠 Brain MRI Tumor Classifier</h1>
          <p>Upload a brain MRI scan and get a tumor-type prediction explained with Grad-CAM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer-banner">⚠️ <strong>Educational and research use only</strong> '
        "— this app is not a medical device. Do not use these predictions for "
        "diagnosis or treatment decisions.</div>",
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`. Cannot run the app.")
        st.stop()

    image = render_input_section()
    if image is None:
        st.info("👆 Upload an MRI or click a sample to begin.")
        return

    render_results(image)


if __name__ == "__main__":
    main()
