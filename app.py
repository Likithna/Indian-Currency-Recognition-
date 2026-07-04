import json
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from security_features import get_security_info

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "currency_model.keras"      
CLASS_NAMES_PATH = "class_names.json"    

# Set wide layout for a dashboard-style look
st.set_page_config(page_title="Currency Recognition System", page_icon="💵", layout="wide")

# --------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------
# The visual identity borrows from the note itself: a deep "security-ink"
# background, a color-shifting green->blue accent (the same shift the real
# RBI security thread makes when a note is tilted), and a monospace face
# for anything numeric — echoing a note's serial-number panel.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink-900:#0C0F14; --ink-850:#12161D; --ink-800:#161B24;
        --line:#262E3B; --line-soft:#1D2430;
        --text-hi:#F1EFE6; --text-lo:#8A93A6; --text-mid:#B7BECC;
        --thread-a:#22A876; --thread-b:#3E7BFF; --brass:#C9A467;
        --warn:#E0A93A;
    }

    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: var(--ink-900) !important;
        color: var(--text-hi);
        font-family: 'IBM Plex Sans', sans-serif;
    }
    [data-testid="stHeader"] { background-color: transparent; }
    .block-container { padding-top: 2.2rem; max-width: 1100px; }

    /* ---- Header ---- */
    .thread-bar {
        height: 4px; width: 100%; border-radius: 2px; margin-bottom: 1.6rem;
        background: linear-gradient(90deg, var(--thread-a), var(--thread-b), var(--thread-a));
        background-size: 220% 100%;
        animation: thread-shift 9s ease-in-out infinite;
    }
    @keyframes thread-shift {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 0%; }
        100% { background-position: 0% 0%; }
    }
    .svs-eyebrow {
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.18em;
        text-transform: uppercase; font-size: 0.72rem; color: var(--brass);
        margin-bottom: 0.4rem;
    }
    .svs-title {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.5rem;
        color: var(--text-hi); margin: 0 0 0.35rem 0; letter-spacing: 0.005em;
    }
    .svs-subtitle {
        font-family: 'IBM Plex Sans'; color: var(--text-lo); font-size: 0.95rem;
        max-width: 640px; line-height: 1.55; margin-bottom: 1.8rem;
    }

    /* ---- Section labels ---- */
    .svs-section-label {
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.14em;
        text-transform: uppercase; font-size: 0.72rem; color: var(--text-lo);
        border-bottom: 1px solid var(--line); padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--ink-850) !important;
        border: 1px dashed var(--line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section { background-color: transparent; }
    [data-testid="stImage"] img {
        border-radius: 10px; border: 1px solid var(--line);
    }

    /* ---- Result card ---- */
    .svs-card {
        background-color: var(--ink-850); border: 1px solid var(--line);
        border-radius: 12px; padding: 1.4rem 1.5rem; margin-bottom: 1.2rem;
    }
    .svs-result-row {
        display: flex; justify-content: space-between; align-items: baseline;
        margin-bottom: 1rem;
    }
    .svs-denom {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.1rem;
        color: var(--text-hi);
    }
    .svs-confidence-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
        color: var(--thread-a); font-weight: 600;
    }
    .svs-confidence-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-lo);
    }
    .svs-meter-track {
        width: 100%; height: 8px; background: var(--ink-800);
        border-radius: 4px; overflow: hidden; border: 1px solid var(--line-soft);
        margin-top: 0.3rem;
    }
    .svs-meter-fill {
        height: 100%; border-radius: 4px;
        background: linear-gradient(90deg, var(--thread-a), var(--thread-b));
    }

    /* ---- Top-3 bars ---- */
    .svs-bar-row { margin-bottom: 0.65rem; }
    .svs-bar-label {
        display: flex; justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        color: var(--text-mid); margin-bottom: 0.25rem;
    }
    .svs-bar-track {
        width: 100%; height: 6px; background: var(--ink-800);
        border-radius: 3px; overflow: hidden;
    }
    .svs-bar-fill { height: 100%; border-radius: 3px; background: var(--brass); }

    /* ---- Dossier stat grid ---- */
    .svs-dossier {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
        margin-bottom: 1.1rem;
    }
    .svs-stat-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-lo);
        margin-bottom: 0.25rem;
    }
    .svs-stat-value {
        font-family: 'IBM Plex Sans'; font-size: 1rem; font-weight: 600;
        color: var(--text-hi);
    }
    .svs-motif {
        font-family: 'IBM Plex Sans'; font-size: 0.9rem; color: var(--text-mid);
        border-top: 1px solid var(--line); padding-top: 0.9rem; margin-top: 0.2rem;
    }

    /* ---- Checklist ---- */
    .svs-check-row {
        display: flex; gap: 0.7rem; align-items: flex-start;
        padding: 0.6rem 0; border-bottom: 1px solid var(--line-soft);
        font-size: 0.88rem; color: var(--text-mid); line-height: 1.5;
    }
    .svs-check-row:last-child { border-bottom: none; }
    .svs-check-mark {
        flex-shrink: 0; width: 18px; height: 18px; border-radius: 50%;
        background: rgba(34, 168, 118, 0.15); color: var(--thread-a);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700; margin-top: 0.1rem;
    }

    /* ---- Withdrawal / alert banner ---- */
    .svs-alert {
        display: flex; gap: 0.75rem; align-items: flex-start;
        background: rgba(224, 169, 58, 0.1); border: 1px solid rgba(224, 169, 58, 0.4);
        border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 1.2rem;
        font-size: 0.88rem; color: var(--text-hi); line-height: 1.5;
    }
    .svs-alert-icon { font-size: 1.15rem; color: var(--warn); flex-shrink: 0; }

    hr { border-color: var(--line) !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image):
    image = image.convert('RGB').resize(IMG_SIZE)
    # MobileNetV3Large (used in the training notebook) has its own built-in
    # Rescaling layer and expects RAW pixel values in [0, 255] — do NOT
    # divide by 255 here, or every image collapses to nearly the same
    # tensor and the model predicts the same class for everything.
    img_array = np.array(image).astype('float32')
    return np.expand_dims(img_array, axis=0)

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown('<div class="thread-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="svs-eyebrow">Capstone Project — Samsung Innovation Campus AIML</div>', unsafe_allow_html=True)
st.markdown('<div class="svs-title">💵 Indian Currency Recognition System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="svs-subtitle">Upload a photo of a banknote to identify its denomination, '
    'then verify it yourself against the official RBI security-feature checklist.</div>',
    unsafe_allow_html=True
)

model, class_names = load_model_and_labels()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="svs-section-label">01 — Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a banknote photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        img_tensor = preprocess_image(image)
        preds = model.predict(img_tensor, verbose=0)[0]
        
        top_idx = int(np.argmax(preds))
        predicted_label = class_names[top_idx]
        confidence = float(preds[top_idx])

        st.markdown('<div class="svs-section-label">02 — Analysis Results</div>', unsafe_allow_html=True)

        # Build the prediction graph — a horizontal bar chart across every
        # class (not just top-3), styled to match the app's palette, and
        # embedded as an image inside our HTML card for consistent styling.
        sorted_idx = np.argsort(preds)[::-1]
        sorted_labels = [f"₹{class_names[i]}" for i in sorted_idx]
        sorted_values = [float(preds[i]) * 100 for i in sorted_idx]
        bar_colors = ["#22A876" if i == top_idx else "#3E7BFF" for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, max(1.8, 0.55 * len(class_names))))
        fig.patch.set_facecolor("#12161D")
        ax.set_facecolor("#12161D")
        ax.barh(sorted_labels[::-1], sorted_values[::-1], color=bar_colors[::-1], height=0.55)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", color="#8A93A6", fontsize=9)
        ax.tick_params(colors="#B7BECC", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#262E3B", linewidth=0.6)
        ax.set_axisbelow(True)
        for i, v in enumerate(sorted_values[::-1]):
            ax.text(v + 2, i, f"{v:.1f}%", va="center", color="#F1EFE6",
                     fontsize=9, fontfamily="monospace")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        graph_b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(f"""
        <div class="svs-card">
            <div class="svs-result-row">
                <div>
                    <div class="svs-confidence-label">Predicted Denomination</div>
                    <div class="svs-denom">₹{predicted_label}</div>
                </div>
                <div style="text-align:right;">
                    <div class="svs-confidence-label">Confidence</div>
                    <div class="svs-confidence-value">{confidence*100:.1f}%</div>
                </div>
            </div>
            <div class="svs-meter-track"><div class="svs-meter-fill" style="width:{confidence*100:.1f}%"></div></div>
        </div>
        <div class="svs-card">
            <div class="svs-confidence-label" style="margin-bottom:0.7rem;">Prediction Graph</div>
            <img src="data:image/png;base64,{graph_b64}" style="width:100%; border-radius:8px;" />
        </div>
        """, unsafe_allow_html=True)

st.divider()

if uploaded_file:
    info = get_security_info(predicted_label)
    if info:
        base_colour = info.get("base_colour", "N/A")
        size_mm = info.get("size_mm", "N/A")
        id_mark = info.get("identification_mark", "N/A")
        reverse_motif = info.get("reverse_motif", "N/A")
        common_features = info.get("common_features", [])
        withdrawal_note = info.get("note")

        checklist_html = "".join(
            f'<div class="svs-check-row"><div class="svs-check-mark">✓</div><div>{feature}</div></div>'
            for feature in common_features
        )

        alert_html = ""
        if withdrawal_note:
            alert_html = f"""
            <div class="svs-alert">
                <div class="svs-alert-icon">⚠</div>
                <div><strong>This denomination has been withdrawn from circulation.</strong><br>{withdrawal_note}</div>
            </div>
            """

        st.markdown('<div class="svs-section-label">03 — Official RBI Security Features</div>', unsafe_allow_html=True)
        st.markdown(f"""
        {alert_html}
        <div class="svs-card">
            <div class="svs-dossier">
                <div>
                    <div class="svs-stat-label">Base Colour</div>
                    <div class="svs-stat-value">{base_colour}</div>
                </div>
                <div>
                    <div class="svs-stat-label">Size (mm)</div>
                    <div class="svs-stat-value">{size_mm}</div>
                </div>
                <div>
                    <div class="svs-stat-label">Identification Mark</div>
                    <div class="svs-stat-value">{id_mark}</div>
                </div>
            </div>
            <div class="svs-motif"><strong>Reverse Motif:</strong> {reverse_motif}</div>
        </div>
        <div class="svs-card">
            <div class="svs-confidence-label" style="margin-bottom:0.5rem;">Checklist to Verify Authenticity</div>
            {checklist_html}
        </div>
        """, unsafe_allow_html=True)
