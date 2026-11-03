"""
Streamlit Web Application for Hand Gesture Recognition with Grad-CAM Explainability.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import streamlit as st
import torch
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import load_model
from src.dataset import CLASSES
try:
    from utils import predict_and_explain
except ImportError:
    from app.utils import predict_and_explain

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Recognition SOTA & XAI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gesture Emojis
GESTURE_EMOJIS = {
    "like": "👍",
    "palm": "✋",
    "peace": "✌️"
}

# Cache model loading to avoid reloading on each interaction
@st.cache_resource
def get_model(model_name: str, weights_path: str, device: str):
    return load_model(
        model_name=model_name,
        weights_path=weights_path if os.path.exists(weights_path) else None,
        num_classes=len(CLASSES),
        device=device
    )


def main():
    st.title("🖐️ Hand Gesture Recognition with SOTA Vision & Grad-CAM")
    st.markdown(
        "Interactive Computer Vision demo featuring modern **SOTA architectures** (ConvNeXt-Tiny) "
        "compared against the **MobileNetV2 baseline**, augmented with **Grad-CAM visual explainability** "
        "and real-time inference latency tracking."
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Compute Device: **{device.upper()}**")

    model_choice = st.sidebar.selectbox(
        "Select Model Architecture:",
        options=["ConvNeXt-Tiny (SOTA)", "MobileNetV2 (Baseline)", "Side-by-Side Comparison"]
    )

    cam_alpha = st.sidebar.slider(
        "Grad-CAM Overlay Opacity (α):",
        min_value=0.2,
        max_value=0.8,
        value=0.55,
        step=0.05,
        help="Controls the blend ratio between the Grad-CAM heatmap and original image."
    )

    # Weights paths
    mobilenet_weights = str(ROOT_DIR / "weights" / "baseline_mobilenet.pth")
    convnext_weights = str(ROOT_DIR / "weights" / "sota_convnext.pth")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    st.sidebar.write(f"- Baseline weights: `{'Found' if os.path.exists(mobilenet_weights) else 'Missing'}`")
    st.sidebar.write(f"- SOTA weights: `{'Found' if os.path.exists(convnext_weights) else 'Missing'}`")

    # Main input section
    input_method = st.radio("Choose Input Source:", ["Upload Image", "Use Webcam"], horizontal=True)

    input_image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a gesture image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Take a snapshot of your hand gesture")
        if camera_file is not None:
            input_image = Image.open(camera_file)

    if input_image is None:
        st.info("👆 Please upload an image or take a photo with your webcam to begin.")
        return

    st.markdown("---")

    # Single model prediction
    if model_choice in ["ConvNeXt-Tiny (SOTA)", "MobileNetV2 (Baseline)"]:
        is_sota = "ConvNeXt" in model_choice
        m_name = "convnext" if is_sota else "mobilenet"
        w_path = convnext_weights if is_sota else mobilenet_weights

        model, cam_layer = get_model(m_name, w_path, device)
        results = predict_and_explain(model, cam_layer, input_image, device=device, cam_alpha=cam_alpha)

        # Metric cards
        col_m1, col_m2, col_m3 = st.columns(3)
        emoji = GESTURE_EMOJIS.get(results["predicted_label"], "")
        col_m1.metric("Predicted Gesture", f"{emoji} {results['predicted_label'].upper()}")
        col_m2.metric("Confidence Score", f"{results['confidence'] * 100:.1f}%")
        col_m3.metric("Latency / Device", f"{results['latency_ms']:.1f} ms ({device.upper()})")

        # Side-by-side visualization
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📷 Original Input")
            st.image(input_image, use_container_width=True)

        with col_right:
            st.subheader("🔥 Grad-CAM Attention Heatmap")
            st.image(results["overlay_rgb"], use_container_width=True)
            st.caption("Warm regions (red/yellow) indicate where the model is looking to classify the gesture.")

        # Probability distribution
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({
            "Gesture": [f"{GESTURE_EMOJIS.get(c, '')} {c.capitalize()}" for c in CLASSES],
            "Probability": [results["probabilities"][c] for c in CLASSES]
        })
        st.bar_chart(prob_df.set_index("Gesture"))

    # Side-by-side comparison mode
    else:
        st.subheader("⚡ Side-by-Side Model Comparison")
        col_base, col_sota = st.columns(2)

        # Baseline
        with col_base:
            st.markdown("### 🔹 Baseline: MobileNetV2")
            base_model, base_layer = get_model("mobilenet", mobilenet_weights, device)
            base_res = predict_and_explain(base_model, base_layer, input_image, device=device, cam_alpha=cam_alpha)

            st.write(f"**Prediction:** {GESTURE_EMOJIS.get(base_res['predicted_label'], '')} `{base_res['predicted_label'].upper()}`")
            st.write(f"**Confidence:** `{base_res['confidence']*100:.1f}%`")
            st.write(f"**Latency:** `{base_res['latency_ms']:.1f} ms`")
            st.image(base_res["overlay_rgb"], caption="MobileNetV2 Grad-CAM", use_container_width=True)

        # SOTA
        with col_sota:
            st.markdown("### 🚀 SOTA: ConvNeXt-Tiny")
            sota_model, sota_layer = get_model("convnext", convnext_weights, device)
            sota_res = predict_and_explain(sota_model, sota_layer, input_image, device=device, cam_alpha=cam_alpha)

            st.write(f"**Prediction:** {GESTURE_EMOJIS.get(sota_res['predicted_label'], '')} `{sota_res['predicted_label'].upper()}`")
            st.write(f"**Confidence:** `{sota_res['confidence']*100:.1f}%`")
            st.write(f"**Latency:** `{sota_res['latency_ms']:.1f} ms`")
            st.image(sota_res["overlay_rgb"], caption="ConvNeXt-Tiny Grad-CAM", use_container_width=True)


if __name__ == "__main__":
    main()

