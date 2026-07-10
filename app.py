import json
import io
import base64
import os
import sys
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from security_features import get_security_info
from rbi_chatbot import get_chatbot_response, is_gemini_configured, identify_note_for_speech, verify_is_currency_note

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "currency_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# A hard-but-genuine photo (distant, rotated, cluttered background) can
# score lower confidence than a confidently-wrong guess on something
# unrelated — that's a real limit of softmax confidence, not something a
# threshold alone fully fixes. Lowered from 0.70, and paired with a margin
# check below (how far the top guess leads the runner-up) so it's not
# just a looser single number.
OOD_CONFIDENCE_THRESHOLD = 0.45
OOD_MARGIN_THRESHOLD = 0.12

st.set_page_config(page_title="Currency Recognition System", page_icon="💵", layout="wide")

# Suppress TF warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------
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
    .block-container { padding-top: 2.2rem; max-width: 1440px; }

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

    .svs-section-label {
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.14em;
        text-transform: uppercase; font-size: 0.72rem; color: var(--text-lo);
        border-bottom: 1px solid var(--line); padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--ink-850) !important;
        border: 1px dashed var(--line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section { background-color: transparent; }
    [data-testid="stImage"] img {
        border-radius: 10px; border: 1px solid var(--line);
    }

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

    .svs-alert {
        display: flex; gap: 0.75rem; align-items: flex-start;
        background: rgba(224, 169, 58, 0.1); border: 1px solid rgba(224, 169, 58, 0.4);
        border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 1.2rem;
        font-size: 0.88rem; color: var(--text-hi); line-height: 1.5;
    }
    .svs-alert-icon { font-size: 1.15rem; color: var(--warn); flex-shrink: 0; }

    .svs-reject-title {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.5rem;
        color: var(--warn); margin: 0.2rem 0 0.5rem 0;
    }
    .svs-reject-body {
        font-family: 'IBM Plex Sans'; font-size: 0.9rem; color: var(--text-mid);
        line-height: 1.55;
    }

    .svs-chat-intro {
        font-family: 'IBM Plex Sans'; font-size: 0.85rem; color: var(--text-lo);
        line-height: 1.5; margin-bottom: 1rem;
    }
    [data-testid="stChatMessage"] {
        background-color: var(--ink-850) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: var(--ink-850) !important;
        color: var(--text-hi) !important;
        border: 1px solid var(--line) !important;
    }
    .svs-chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }

    hr { border-color: var(--line) !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_labels():
    """Load the trained model and class names with error handling."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'currency_model.keras' and 'class_names.json' are in the repo.")
        raise

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image):
    image = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(image).astype('float32')
    return np.expand_dims(img_array, axis=0)

def speak_text(text: str, key: str = "tts"):
    """
    Speaks text aloud using the browser's built-in Web Speech API — free,
    no extra API call needed for the audio itself, works in all major
    browsers. Triggered only from a button click (a real user gesture),
    which satisfies browsers' autoplay restrictions on speech synthesis.
    """
    safe = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    components.html(f"""
        <script>
        (function() {{
            try {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{safe}");
                msg.rate = 0.95;
                window.speechSynthesis.speak(msg);
            }} catch (e) {{ console.error("Speech synthesis failed:", e); }}
        }})();
        </script>
    """, height=0)

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown('<div class="thread-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="svs-eyebrow">Capstone Project — Samsung Innovation Campus AIML</div>', unsafe_allow_html=True)
st.markdown('<div class="svs-title">💵 Indian Currency Recognition System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="svs-subtitle">Upload a photo of a banknote to identify its denomination, '
    'verify it against the official RBI security-feature checklist, and ask the '
    'assistant on the right about currency rules and guidelines.</div>',
    unsafe_allow_html=True
)

# Check if model files exist
model_exists = os.path.exists(MODEL_PATH)
labels_exists = os.path.exists(CLASS_NAMES_PATH)

if not model_exists or not labels_exists:
    st.error("⚠️ Model files not found!")
    st.markdown(f"""
    <div class="svs-card">
        <p>Please ensure the following files are in your repository:</p>
        <ul>
            <li><code>currency_model.keras</code> — {"✅ Found" if model_exists else "❌ Missing"}</li>
            <li><code>class_names.json</code> — {"✅ Found" if labels_exists else "❌ Missing"}</li>
        </ul>
        <p>Run the training notebook in Google Colab to generate these files.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    model, class_names = load_model_and_labels()
except Exception:
    st.stop()

col1, col2, col3 = st.columns([1, 1.4, 1.1])

image = None

with col1:
    st.markdown('<div class="svs-section-label">01 — Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a banknote photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

is_confident = False
predicted_label = None

with col2:
    if uploaded_file:
        img_tensor = preprocess_image(image)
        preds = model.predict(img_tensor, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        predicted_label = class_names[top_idx]
        confidence = float(preds[top_idx])

        # OOD check, stage 1: absolute confidence alone has real limits —
        # a hard but genuine photo can score lower than a confidently-wrong
        # guess on something unrelated. A margin check (how far ahead the
        # top guess is over the runner-up) catches more genuine-but-hard
        # notes without fully reopening the door to unrelated images.
        sorted_probs = np.sort(preds)[::-1]
        second_best = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = confidence - second_best
        is_confident = confidence >= OOD_CONFIDENCE_THRESHOLD and margin >= OOD_MARGIN_THRESHOLD

        # OOD check, stage 2: confidence+margin still has a blind spot —
        # the CNN was only ever trained to pick among 11 currency classes,
        # so it can be BOTH highly confident AND have a wide margin on
        # something it's never seen (e.g. a person's photo), since it has
        # no concept of "none of the above". If stage 1 passed, ask Gemini
        # the actual question directly: is this really a currency note?
        gemini_overrode_rejection = False
        if is_confident and is_gemini_configured():
            with st.spinner("Verifying with an independent AI check..."):
                gemini_says_note, _verify_err = verify_is_currency_note(image)
            if gemini_says_note is False:
                is_confident = False
                gemini_overrode_rejection = True
            # if gemini_says_note is None (call failed/no key/ambiguous),
            # fail open and trust the CNN's own result rather than block
            # the whole app on a flaky network call.

        st.markdown('<div class="svs-section-label">02 — Analysis Results</div>', unsafe_allow_html=True)

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

        if is_confident:
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
            """, unsafe_allow_html=True)
        else:
            if gemini_overrode_rejection:
                st.markdown(f"""
                <div class="svs-card" style="border-color: var(--warn);">
                    <div class="svs-confidence-label">Result</div>
                    <div class="svs-reject-title">⚠ Not Recognized as an Indian Currency Note</div>
                    <div class="svs-reject-body">
                        The image classifier guessed ₹{predicted_label} ({confidence*100:.1f}% confidence),
                        but an independent AI vision check confirmed this photo does
                        <strong>not</strong> actually show a currency note. Try a clear,
                        well-lit, close-up photo of a single note instead.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="svs-card" style="border-color: var(--warn);">
                    <div class="svs-confidence-label">Result</div>
                    <div class="svs-reject-title">⚠ Not Recognized as an Indian Currency Note</div>
                    <div class="svs-reject-body">
                        The model's best guess (₹{predicted_label}, {confidence*100:.1f}% confidence,
                        {margin*100:.1f} points ahead of the runner-up) didn't clear the bar needed
                        to trust a result (≥{OOD_CONFIDENCE_THRESHOLD*100:.0f}% confidence AND
                        ≥{OOD_MARGIN_THRESHOLD*100:.0f} points ahead of the next guess).
                        This usually means the photo isn't a currency note, or the note
                        isn't clearly visible. Try a clear, well-lit, close-up photo of
                        a single note.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="svs-card">
            <div class="svs-confidence-label" style="margin-bottom:0.7rem;">Prediction Graph</div>
            <img src="data:image/png;base64,{graph_b64}" style="width:100%; border-radius:8px;" />
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Accessibility — read the note aloud for visually impaired users.
        # Deliberately independent of the CNN/OOD result above: this asks
        # Gemini to look at the photo itself, so it works (and is honest)
        # even when the app's own model was uncertain.
        # ------------------------------------------------------------------
        st.markdown('<div class="svs-confidence-label" style="margin:0.3rem 0 0.6rem;">Accessibility</div>', unsafe_allow_html=True)
        if st.button("🔊 Read note aloud (independent AI check)", key="speak_btn", use_container_width=True):
            if is_gemini_configured():
                with st.spinner("Asking Gemini to look at the note..."):
                    spoken_text, speak_error = identify_note_for_speech(image)
                if spoken_text:
                    st.markdown(f'<div class="svs-card">🔊 "{spoken_text}"</div>', unsafe_allow_html=True)
                    speak_text(spoken_text)
                else:
                    st.warning(f"Live AI check failed, so nothing was read aloud. Debug info: {speak_error}")
            else:
                st.warning(
                    "This needs a Gemini API key configured (see README) for "
                    "an independent check. Reading the app's own prediction "
                    "instead, as a fallback:"
                )
                _denom_words = {
                    "10": "Ten Rupees", "20": "Twenty Rupees", "50": "Fifty Rupees",
                    "100": "Hundred Rupees", "200": "Two Hundred Rupees",
                    "500": "Five Hundred Rupees", "2000": "Two Thousand Rupees",
                }
                fallback_text = "Not a currency note"
                if is_confident:
                    _info = get_security_info(predicted_label)
                    _denom_key = _info["denomination"] if _info else None
                    fallback_text = _denom_words.get(_denom_key, f"{predicted_label} Rupees")
                st.markdown(f'<div class="svs-card">🔊 "{fallback_text}"</div>', unsafe_allow_html=True)
                speak_text(fallback_text)

st.divider()

if uploaded_file and is_confident:
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

# --------------------------------------------------------------------------
# 04 — RBI Guidelines Assistant (right-side chatbot panel)
# --------------------------------------------------------------------------
with col3:
    st.markdown('<div class="svs-section-label">04 — RBI Guidelines Assistant</div>', unsafe_allow_html=True)

    _last_error = st.session_state.get("last_chat_error")
    if is_gemini_configured() and not (_last_error and _last_error != "no_api_key"):
        status_html = '<span style="color:var(--thread-a);">● Live AI (Gemini)</span>'
    elif is_gemini_configured():
        status_html = '<span style="color:var(--warn);">● Key set, but last call failed — see debug info below</span>'
    else:
        status_html = '<span style="color:var(--warn);">● Offline mode — no API key set</span>'

    st.markdown(
        f'<div class="svs-chat-intro">Ask anything — general knowledge or '
        f'RBI-specific — with a focus on currency rules and guidelines. '
        f'{status_html}</div>',
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hi! Ask me anything — general knowledge or "
                           "currency-specific, like \"is the 2000 note "
                           "still legal tender?\" or \"what do I do with "
                           "a fake note?\"",
            }
        ]

    suggested_questions = [
        "Is the 2000 note still legal tender?",
        "What if I get a fake note?",
        "How do I exchange a torn note?",
        "Where is Indian currency printed?",
    ]

    chat_container = st.container(height=340)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    last_error = st.session_state.get("last_chat_error")
    if last_error and last_error != "no_api_key":
        with st.expander("⚠ Live AI call failed — debug info"):
            st.code(last_error)
            st.caption(
                "Common causes: API key not set in this deployment's "
                "Secrets, key pasted with extra spaces/quotes, free-tier "
                "rate limit hit (many requests in a short time), or a "
                "network restriction. Share this exact message if you "
                "need help diagnosing it further."
            )

    chip_cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        if chip_cols[i % 2].button(question, key=f"chip_{i}", use_container_width=True):
            history_so_far = list(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": question})
            answer, _source, error_detail = get_chatbot_response(
                question, history=history_so_far, image=image
            )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.last_chat_error = error_detail
            st.rerun()

    user_question = st.chat_input("Ask about RBI guidelines or anything else...")
    if user_question:
        history_so_far = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        answer, _source, error_detail = get_chatbot_response(
            user_question, history=history_so_far, image=image
        )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.last_chat_error = error_detail
        st.rerun()
