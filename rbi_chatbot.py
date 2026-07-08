"""
rbi_chatbot.py
--------------
The RBI Guidelines Assistant. Primary path: calls Google's Gemini API for
real, general-purpose answers (not limited to a fixed FAQ list). Now with
VISION SUPPORT — the chatbot can see the uploaded banknote image and
respond accordingly.

Fallback path: if no API key is configured, or the request fails for any reason
(offline, rate-limited, key revoked), it automatically drops back to a
small offline FAQ system (TF-IDF + cosine similarity over a curated RBI
knowledge base) so the app never just breaks mid-demo.

Why Gemini specifically: Google AI Studio's Gemini API has a genuinely
free tier (no credit card, doesn't expire) for its "Flash" model family.
Get a free key at https://aistudio.google.com/apikey.
"""

import os
import io
import base64
import requests
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Gemini API (primary path)
# --------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = (
    "You are the 'RBI Guidelines Assistant', embedded inside an Indian "
    "Currency Recognition System — a student capstone project. You can "
    "answer general-knowledge questions on any topic, but you specialize "
    "in RBI (Reserve Bank of India) rules, legal tender, counterfeit "
    "notes, currency exchange rules, and banknote security features. "
    "Keep answers concise (2-5 sentences) unless the user asks for more "
    "detail. If you're not confident about a specific rule or figure "
    "(especially anything that could have changed recently, like "
    "withdrawal deadlines), say so and suggest checking rbi.org.in."
)

VISION_SYSTEM_PROMPT = (
    "You are the 'RBI Guidelines Assistant' with VISION capabilities, "
    "embedded inside an Indian Currency Recognition System. The user has "
    "uploaded a photo of an Indian banknote. You can see this image and "
    "answer questions about it specifically — for example, identify "
    "security features visible in the photo, assess image quality, "
    "point out what to look for, or answer general RBI questions. "
    "Keep answers concise (2-5 sentences) unless asked for more detail. "
    "If you cannot clearly see something in the image, say so honestly."
)


def _get_api_key():
    """Reads the key from Streamlit secrets first, then env vars."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def _pil_to_base64(image: Image.Image, fmt="JPEG") -> str:
    """Convert a PIL image to a base64 string for Gemini inlineData."""
    buf = io.BytesIO()
    # Convert to RGB if necessary (e.g. RGBA -> RGB)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_gemini(query: str, history: list | None = None, image: Image.Image | None = None) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY configured")

    # Build contents array with history
    contents = []
    for msg in (history or [])[-8:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Build the current user message parts
    parts = []

    # Add image if provided
    if image is not None:
        b64_data = _pil_to_base64(image, fmt="JPEG")
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64_data
            }
        })

    # Add text prompt
    parts.append({"text": query})
    contents.append({"role": "user", "parts": parts})

    # Choose system prompt based on whether image is present
    system_prompt = VISION_SYSTEM_PROMPT if image is not None else SYSTEM_PROMPT

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
    }

    response = requests.post(
        GEMINI_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,  # slightly longer for image uploads
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# --------------------------------------------------------------------------
# Offline FAQ fallback (TF-IDF + cosine similarity)
# --------------------------------------------------------------------------
RBI_KNOWLEDGE_BASE = [
    {
        "q": "What is legal tender?",
        "a": "Legal tender means a currency that must be accepted if offered "
             "in payment of a debt. Every banknote issued by the RBI carries "
             "the Governor's signature and the words 'I promise to pay the "
             "bearer the sum of...' — this is the RBI's guarantee, and it's "
             "what makes the note legal tender under Section 26 of the RBI "
             "Act, 1934.",
    },
    {
        "q": "Who issues banknotes in India?",
        "a": "The Reserve Bank of India is the sole authority for issuing "
             "banknotes in India, under Section 22 of the RBI Act, 1934. "
             "The one exception is the ₹1 note/coin, which is issued by the "
             "Government of India and signed by the Finance Secretary, not "
             "the RBI Governor.",
    },
    {
        "q": "Is the 2000 rupee note still legal tender?",
        "a": "Yes. The RBI withdrew ₹2000 notes from circulation starting "
             "May 2023 (printing had already stopped back in 2018-19), but "
             "they remain legal tender — you just won't see them handed out "
             "as change anymore. Check rbi.org.in for the latest exchange "
             "arrangements, since these have been updated over time.",
    },
    {
        "q": "What happened to the 500 and 1000 rupee notes in 2016?",
        "a": "On 8 November 2016, the Government of India demonetised the "
             "old ₹500 and ₹1000 notes — they stopped being legal tender "
             "overnight. New ₹500 and ₹2000 notes in the Mahatma Gandhi "
             "(New) Series were introduced shortly after, followed by "
             "redesigned ₹10, ₹20, ₹50, ₹100 and ₹200 notes over the "
             "following years.",
    },
    {
        "q": "Where are Indian banknotes printed?",
        "a": "At four presses. Two are owned by the Government of India "
             "via SPMCIL: the Currency Note Press in Nashik (Maharashtra) "
             "and the Bank Note Press in Dewas (Madhya Pradesh). The other "
             "two are owned by the RBI's own subsidiary, BRBNMPL, at "
             "Mysuru (Karnataka) and Salboni (West Bengal).",
    },
    {
        "q": "Where are Indian coins minted?",
        "a": "Coins are minted at four government-owned mints: Mumbai, "
             "Hyderabad, Kolkata and Noida, run by SPMCIL. The Government "
             "of India designs and mints coins under the Coinage Act, "
             "2011, and the RBI puts them into circulation under Section "
             "38 of the RBI Act.",
    },
    {
        "q": "What should I do if I receive a fake or counterfeit note?",
        "a": "Don't try to spend it or pass it on — knowingly using a "
             "counterfeit note is an offence. Take it to the nearest bank "
             "branch or police station. Banks are required to impound "
             "suspected forged notes and issue you an acknowledgement "
             "receipt. There's no refund for a note confirmed as "
             "counterfeit, since it never had real value — which is "
             "exactly why checking the security features before accepting "
             "a note matters.",
    },
    {
        "q": "Can I exchange a torn or mutilated note?",
        "a": "Yes. Under the RBI (Note Refund) Rules, 2009 (amended 2018), "
             "all bank branches must accept and evaluate mutilated or "
             "imperfect notes. Soiled notes are exchanged for full value; "
             "for mutilated notes, the refund depends on how much of the "
             "original note is present — you may get full or half value "
             "depending on the surviving area.",
    },
    {
        "q": "What is the Clean Note Policy?",
        "a": "It's an RBI initiative (introduced in 2001) to keep only "
             "good-quality, genuine notes in circulation. It covers things "
             "like banks not stapling note packets, sorting notes into "
             "reissuable vs. non-issuable, and periodically withdrawing "
             "and destroying soiled or unfit notes.",
    },
    {
        "q": "What are the current Indian banknote denominations?",
        "a": "Notes are currently issued in ₹10, ₹20, ₹50, ₹100, ₹200, and "
             "₹500 in the Mahatma Gandhi (New) Series. ₹2000 notes are no "
             "longer issued but remain legal tender.",
    },
    {
        "q": "What is a watermark on a banknote?",
        "a": "It's a security feature built into the paper itself — hold "
             "the note up to light and you'll see a faint, shaded portrait "
             "of Mahatma Gandhi along with the denomination numeral, "
             "visible in the blank space to the left of the main design.",
    },
    {
        "q": "What is the security thread on a banknote?",
        "a": "A thin embedded thread running through the note, inscribed "
             "with 'भारत' and 'RBI'. It appears in patches on the surface "
             "and shifts colour from green to blue when you tilt the note "
             "— that colour-shift is one of the harder features to fake "
             "convincingly.",
    },
    {
        "q": "What is intaglio printing?",
        "a": "A printing technique where ink is applied with enough "
             "pressure that it sits slightly raised off the paper. On "
             "Indian banknotes, Gandhi's portrait, the Ashoka Pillar "
             "emblem, the RBI seal and the guarantee clause are all "
             "printed this way — you can feel the texture with a "
             "fingertip, which also helps visually impaired users identify "
             "notes.",
    },
    {
        "q": "What is the identification mark on a banknote for the blind?",
        "a": "Each denomination (except ₹10) has a unique raised tactile "
             "shape printed near the watermark — for example a circle on "
             "₹500, a triangle on ₹100 — so visually impaired users can "
             "identify the denomination by touch alone.",
    },
    {
        "q": "What is RBI's MANI app?",
        "a": "MANI (Mobile Aided Note Identifier) is RBI's own app that "
             "helps visually impaired people identify the denomination of "
             "a note using their phone's camera. It doesn't verify "
             "authenticity — just denomination — which is the same "
             "division of labour this project uses: a model identifies "
             "the note, then a checklist helps the user verify it "
             "themselves.",
    },
    {
        "q": "Can a machine or app verify if a note is genuine?",
        "a": "Not reliably from a photo alone — genuine notes are easy to "
             "photograph clearly, so image-based confidence isn't a "
             "trustworthy signal of authenticity by itself. Banks use "
             "dedicated Currency Verification & Processing Systems (CVPS) "
             "hardware that checks multiple physical security features at "
             "once. For individuals, the reliable approach is manually "
             "checking the official security features — watermark, "
             "thread, intaglio printing, etc.",
    },
    {
        "q": "What is currency in circulation?",
        "a": "Currency in Circulation (CiC) refers to the total value of "
             "banknotes and coins held by the public and businesses at any "
             "given time — it's one of the key figures the RBI tracks "
             "when deciding how many new notes to print each year.",
    },
    {
        "q": "What is a currency chest?",
        "a": "A secure storage facility run by select banks on the RBI's "
             "behalf, used to stock and distribute banknotes and coins to "
             "bank branches in their region — it's the layer between the "
             "RBI's central vaults and your local bank branch.",
    },
    {
        "q": "Who designs Indian banknotes?",
        "a": "The RBI's Central Board approves banknote designs, and the "
             "Governor's signature appears on every note. Designs are "
             "periodically updated — most recently with the Mahatma "
             "Gandhi (New) Series introduced after the 2016 "
             "demonetisation.",
    },
    {
        "q": "What is a soiled note versus a mutilated note?",
        "a": "A soiled note is one that's simply dirty from normal wear "
             "and tear (or two halves of the same note pasted together) "
             "but still has all its essential features — banks exchange "
             "these for full value. A mutilated note is torn, cut, or "
             "otherwise damaged so that only part of it survives — its "
             "refund value depends on how much of the note remains.",
    },
    {
        "q": "What is the RBI Act 1934?",
        "a": "The founding legislation that established the Reserve Bank "
             "of India and gives it, among other powers, the sole "
             "authority to issue currency notes in India (Section 22) and "
             "defines what counts as legal tender (Section 26).",
    },
]

_vectorizer = TfidfVectorizer(stop_words="english")
_question_matrix = _vectorizer.fit_transform(
    [item["q"] for item in RBI_KNOWLEDGE_BASE]
)

FALLBACK_UNMATCHED = (
    "I don't have a confident offline answer for that, and the live AI "
    "isn't reachable right now. Try asking about legal tender, "
    "counterfeit notes, mutilated note exchange, or security features."
)


def _offline_answer(query: str, threshold: float = 0.15) -> str:
    if not query or not query.strip():
        return FALLBACK_UNMATCHED
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _question_matrix)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    if best_score < threshold:
        return FALLBACK_UNMATCHED
    return RBI_KNOWLEDGE_BASE[best_idx]["a"]


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def is_gemini_configured() -> bool:
    return bool(_get_api_key())


def get_chatbot_response(query: str, history: list | None = None, image: Image.Image | None = None):
    """
    Returns (answer_text, source, error_detail).
      source is 'gemini' or 'offline'.
      error_detail is None on success, or a short diagnostic string when
      Gemini was configured but the call actually failed (bad key, rate
      limit, network issue, wrong model, etc) — surfaced in the UI so
      failures are debuggable instead of silently hidden.
    """
    if not query or not query.strip():
        return FALLBACK_UNMATCHED, "offline", None

    api_key = _get_api_key()
    if not api_key:
        return _offline_answer(query), "offline", "no_api_key"

    try:
        answer = _call_gemini(query, history, image)
        return answer, "gemini", None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = e.response.text[:300] if e.response is not None else str(e)
        return _offline_answer(query), "offline", f"HTTP {status}: {body}"
    except requests.exceptions.RequestException as e:
        return _offline_answer(query), "offline", f"Network error: {type(e).__name__}: {e}"
    except Exception as e:
        return _offline_answer(query), "offline", f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    # quick self-test (works even without an API key — just exercises the
    # offline path in that case)
    for test_q in [
        "RBI full form?",
        "is 2000 note still valid",
        "what if I get a fake note",
        "explain photosynthesis",
    ]:
        ans, src = get_chatbot_response(test_q)
        print(f"[{src}] {test_q} -> {ans[:80]}...")
