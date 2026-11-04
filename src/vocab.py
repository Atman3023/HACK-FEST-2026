"""The 6-sign vocabulary: display text, typed aliases (fuzzy matched), reference media."""
import difflib
import re

from src import config

DISPLAY = {
    "A": "Letter A", "B": "Letter B", "C": "Letter C",
    "BLOOD_PRESSURE": "Blood pressure check",
    "FEVER": "Fever",
    "STOMACH_ACHE": "Stomach ache",
    "NONE": "Not a known sign",
}

ALIASES = {
    "BLOOD_PRESSURE": ["bp", "b p", "blood pressure", "bloodpressure", "pressure", "bp check",
                       "blood pressure check", "high bp", "low bp", "hypertension"],
    "FEVER": ["fever", "feverish", "temperature", "temp", "high temperature", "bukhar", "bukhaar"],
    "STOMACH_ACHE": ["stomach ache", "stomachache", "stomach pain", "stomach", "tummy ache", "tummy",
                     "belly pain", "abdominal pain", "pet dard", "pet me dard", "pet mein dard"],
}


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def match_action(text):
    """-> {action: score in [0, 1]}. Exact alias (or alias as whole words inside the text) = 1.0,
    otherwise the best difflib similarity to any alias."""
    t = normalize(text)
    scores = {}
    for action, aliases in ALIASES.items():
        best = 0.0
        for a in aliases:
            if t == a or re.search(rf"\b{re.escape(a)}\b", t) and len(a) > 2:
                best = 1.0
                break
            best = max(best, difflib.SequenceMatcher(None, t, a).ratio())
        scores[action] = best
    return scores


def reference_media(cls):
    """Path of the reference image (letter) or clip (action), or None."""
    if cls in config.LETTERS:
        cands = [config.ISL_ASSETS_DIR / f"{cls}{e}" for e in (".png", ".jpg", ".jpeg")]
    else:
        cands = [config.ISL_ASSETS_DIR / "actions" / f"{cls}{e}" for e in (".mp4", ".webm", ".mov")]
    return next((p for p in cands if p.exists()), None)
