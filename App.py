# ============================================================
#  CN-Detect Web App — Streamlit
#  Chest X-ray Disease Classifier
#  Classes: COVID19 | NORMAL | PNEUMONIA | TUBERCULOSIS
# ============================================================

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model import load_model

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "CN-Detect | Chest X-ray Classifier",
    page_icon   = "🩺",
    layout      = "centered",
    initial_sidebar_state = "collapsed"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-covid    { background-color: #FDECEA; border: 2px solid #E53935; }
    .result-normal   { background-color: #E8F5E9; border: 2px solid #43A047; }
    .result-pneumonia{ background-color: #FFF8E1; border: 2px solid #FB8C00; }
    .result-tb       { background-color: #F3E5F5; border: 2px solid #8E24AA; }
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background: #f9f9f9;
        border-radius: 8px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL — cached so it loads only once
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    model, classes, img_size = load_model(
        "cn_detect_effb4_swin_best.pth", device="cpu")
    return model, classes, img_size

# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────
def get_transform(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]),
    ])

# ─────────────────────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────
def predict(image, model, classes, img_size):
    transform = get_transform(img_size)
    tensor    = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]
    pred_idx  = probs.argmax().item()
    pred_class= classes[pred_idx]
    confidence= probs[pred_idx].item() * 100
    all_probs = {cls: probs[i].item() * 100
                 for i, cls in enumerate(classes)}
    return pred_class, confidence, all_probs

# ─────────────────────────────────────────────────────────────
# CLASS COLORS AND EMOJIS
# ─────────────────────────────────────────────────────────────
CLASS_INFO = {
    "COVID19"      : {"emoji": "🦠", "color": "#E53935",
                      "css": "result-covid",
                      "desc": "COVID-19 infection detected"},
    "NORMAL"       : {"emoji": "✅", "color": "#43A047",
                      "css": "result-normal",
                      "desc": "No disease detected — Lungs appear normal"},
    "PNEUMONIA"    : {"emoji": "🫁", "color": "#FB8C00",
                      "css": "result-pneumonia",
                      "desc": "Pneumonia infection detected"},
    "TUBERCULOSIS" : {"emoji": "⚠️", "color": "#8E24AA",
                      "css": "result-tb",
                      "desc": "Tuberculosis signs detected"},
}

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🩺 CN-Detect</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Chest X-ray Disease Classifier — COVID-19 | Normal | Pneumonia | Tuberculosis</p>',
            unsafe_allow_html=True)

# Model info bar
col1, col2, col3 = st.columns(3)
col1.metric("Model", "EffB4 + Swin + CAFS")
col2.metric("Test Accuracy", "95.15%")
col3.metric("Classes", "4")

st.divider()

# Upload section
st.subheader("📤 Upload Chest X-ray Image")
uploaded = st.file_uploader(
    "Accepted formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image for disease classification")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col_res:
        st.subheader("🔍 Analysis")
        with st.spinner("Analysing X-ray..."):
            try:
                model, classes, img_size = get_model()
                pred_class, confidence, all_probs = predict(
                    image, model, classes, img_size)

                info = CLASS_INFO.get(pred_class, {
                    "emoji": "❓", "color": "#333",
                    "css": "result-normal",
                    "desc": pred_class})

                # Result box
                st.markdown(f"""
                <div class="result-box {info['css']}">
                    <h1>{info['emoji']}</h1>
                    <h2 style="color:{info['color']};margin:0">{pred_class}</h2>
                    <p style="margin:4px 0;font-size:0.9rem">{info['desc']}</p>
                    <h3 style="color:{info['color']};margin:4px 0">
                        Confidence: {confidence:.1f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # All class probabilities
                st.subheader("📊 All Class Probabilities")
                for cls, prob in sorted(
                        all_probs.items(), key=lambda x: x[1], reverse=True):
                    ci = CLASS_INFO.get(cls, {"emoji": "❓", "color": "#333"})
                    st.markdown(
                        f'<p class="metric-label">'
                        f'{ci["emoji"]} {cls}</p>',
                        unsafe_allow_html=True)
                    st.progress(prob / 100,
                                text=f"{prob:.1f}%")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check the model file is present.")

st.divider()

# About section
with st.expander("ℹ️ About CN-Detect"):
    st.markdown("""
    **CN-Detect** is a hybrid deep learning model for chest X-ray classification.

    **Architecture:**
    - 🔷 EfficientNet-B4 — extracts local texture features
    - 🔶 Swin-Tiny Transformer — captures global lung structure
    - 🔗 Cross-Attention Feature Selection (CAFS) — novel bidirectional fusion

    **Performance:**
    - Test Accuracy: **95.15%**
    - Tuberculosis Recall: **1.00** (perfect)
    - Pneumonia Recall: **1.00** (perfect)

    **Research:** Final Year Design Project — Daffodil International University

    **Classes:** COVID-19 | Normal | Pneumonia | Tuberculosis
    """)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong>
    This tool is for research and educational purposes only.
    It is NOT a substitute for professional medical diagnosis.
    Always consult a qualified radiologist or physician for clinical decisions.
</div>
""", unsafe_allow_html=True)