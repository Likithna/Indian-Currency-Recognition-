"""
security_features.py
---------------------
Official RBI (Reserve Bank of India) reference data for the Mahatma Gandhi
(New) Series banknotes.

IMPORTANT — read this before you present the project:
This module is a LOOKUP TABLE, not a machine-learning model. It does NOT
detect counterfeits. No public, legal dataset of forged Indian currency
exists for a student (or anyone) to train a classifier on, and no dataset
in the world lets a model verify whether a specific physical note is
genuine just from a photo — genuine notes are far too easy to photograph
well for that to be a meaningful signal.

What this DOES do, and why it satisfies the "anti-counterfeit information"
requirement of the assignment: once the CNN identifies which denomination
a note is, this module returns the OFFICIAL security features a human can
check by eye / touch / UV light to verify that note themselves. This is
exactly the approach RBI uses in its own "MANI" (Mobile Aided Note
Identifier) app for the visually impaired — identify the denomination,
then surface the checklist.

Sources (public, official RBI data): rbi.org.in, paisaboltahai.rbi.org.in,
indiancurrency.rbi.org.in — cross-checked June 2026.
"""

import re

# Features shared by every denomination in the Mahatma Gandhi (New) Series
COMMON_FEATURES = [
    "See-through register: hold the note up to light — the denomination "
    "numeral printed on the front and back should line up and merge into "
    "one clean number.",
    "Watermark: tilt the note — a light-and-shade portrait of Mahatma "
    "Gandhi and the denomination numeral should be visible in the blank "
    "space.",
    "Windowed security thread: a thin thread reading 'भारत' and 'RBI' "
    "that is visible in patches; it shifts colour from green to blue "
    "when the note is tilted.",
    "Latent image: hold the note flat at eye level — a hidden band next "
    "to Gandhi's portrait should reveal the denomination value.",
    "Micro-lettering: tiny text reading 'RBI' and the denomination value, "
    "visible only with a magnifying glass, near the watermark.",
    "Intaglio (raised) printing: Gandhi's portrait, the Ashoka Pillar "
    "emblem, the RBI seal and the guarantee clause are printed with "
    "slightly raised ink you should be able to feel with a fingertip.",
    "Number panel: serial numbers grow left-to-right in size; the panel "
    "and embedded optical fibres glow under UV light.",
    "Language panel: on the reverse, the denomination is written out in "
    "15 of India's 22 official languages.",
]

# Denomination-specific details
SECURITY_FEATURES = {
    "10": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Chocolate Brown",
        "size_mm": "63 x 123",
        "reverse_motif": "Sun Temple, Konark",
        "identification_mark": "None — smallest note, no raised tactile shape",
    },
    "20": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Greenish Yellow",
        "size_mm": "63 x 129",
        "reverse_motif": "Ellora Caves",
        "identification_mark": "Vertical Rectangle (raised, left of the watermark)",
    },
    "50": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Fluorescent Blue",
        "size_mm": "66 x 135",
        "reverse_motif": "Hampi with Chariot",
        "identification_mark": "Square (raised, left of the watermark)",
    },
    "100": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Lavender",
        "size_mm": "66 x 142",
        "reverse_motif": "Rani ki Vav (Queen's Stepwell), Patan",
        "identification_mark": "Triangle (raised, left of the watermark)",
    },
    "200": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Bright Yellow",
        "size_mm": "66 x 146",
        "reverse_motif": "Sanchi Stupa",
        "identification_mark": "H-shaped (raised, left of the watermark)",
    },
    "500": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Stone Grey",
        "size_mm": "66 x 150",
        "reverse_motif": "Red Fort, with Swachh Bharat logo",
        "identification_mark": "Circle (raised, left of the watermark)",
    },
    "2000": {
        "series": "Mahatma Gandhi (New) Series",
        "base_colour": "Magenta",
        "size_mm": "66 x 166",
        "reverse_motif": "Mangalyaan (Mars Orbiter Mission)",
        "identification_mark": "Horizontal Rectangle (raised, left of the watermark)",
        "note": "RBI withdrew ₹2000 notes from circulation from May 2023 "
                "onward. They remain legal tender but are rarely seen day "
                "to day — worth calling out in your app/report.",
    },
}


# Maps the word-based class names used by the current training dataset
# (e.g. 'twenty_old', 'five_hundred', 'two_thousand') to their numeric
# denomination. The old/new suffix is stripped separately below.
_WORD_TO_DENOM = {
    "ten": "10",
    "twenty": "20",
    "fifty": "50",
    "hundred": "100",
    "two_hundred": "200",
    "five_hundred": "500",
    "two_thousand": "2000",
}


def extract_denomination(label: str):
    """
    Pull the rupee denomination out of a raw class/folder name. Handles
    both numeric-style names ('500_new', 'Rs.500', '2000-2016') and the
    word-based names used by the current dataset ('twenty_old',
    'five_hundred', 'two_thousand'). Handles 2000 before 200/100/50/20/10
    so it isn't matched as a substring.

    Returns the denomination as a string ('500') or None if not found.
    """
    normalized = str(label).strip().lower()

    # 1) Numeric-style names
    for denom in ["2000", "500", "200", "100", "50", "20", "10"]:
        if re.search(rf"(?<!\d){denom}(?!\d)", normalized):
            return denom

    # 2) Word-based names — strip a trailing old/new marker, then look up
    cleaned = re.sub(r"[\s_-]*(old|new)$", "", normalized).strip()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    return _WORD_TO_DENOM.get(cleaned)


def get_security_info(denomination) -> dict | None:
    """
    denomination: str or int, e.g. '500' or 500, or a raw class-folder
    name like '500_new' (it will be normalised automatically).
    Returns a dict with denomination-specific info plus the common
    checklist, or None if the denomination isn't recognised.
    """
    key = extract_denomination(str(denomination)) or str(denomination).strip()
    info = SECURITY_FEATURES.get(key)
    if info is None:
        return None
    result = dict(info)
    result["common_features"] = COMMON_FEATURES
    result["denomination"] = key
    return result


if __name__ == "__main__":
    # quick self-test
    for test_label in ["500", "500_new", "Rs.2000", "old_100_note", "20"]:
        info = get_security_info(test_label)
        print(test_label, "->", info["denomination"] if info else None)
