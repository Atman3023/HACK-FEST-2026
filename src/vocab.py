"""The 6-sign vocabulary: display text, typed synonyms (fuzzy matched), reference media."""
import difflib
import re

from src import config

SIGNS = config.LETTERS + config.ACTIONS

DISPLAY = {
    "A": "Letter A", "B": "Letter B", "C": "Letter C",
    "BLOOD_PRESSURE": "Blood pressure check",
    "FEVER": "Fever",
    "STOMACH_ACHE": "Stomach ache",
    "NONE": "Not a known sign",
}

ALIASES = {
    "BLOOD_PRESSURE": ["bp", "b p", "blood pressure", "bloodpressure", "pressure", "bp check", "check bp",
                       "blood pressure check", "check blood pressure", "measure blood pressure", "measure bp",
                       "high bp", "low bp", "hypertension", "bp test"],
    "FEVER": ["fever", "feverish", "temperature", "temp", "high temperature", "feeling hot", "i feel hot",
              "body is hot", "bukhar", "bukhaar", "taap"],
    "STOMACH_ACHE": ["stomach ache", "stomachache", "stomach pain", "stomach", "tummy ache", "tummy pain",
                     "tummy", "belly pain", "abdominal pain", "pet dard", "pet me dard", "pet mein dard",
                     "pet dukh"],
}

FUZZY_MIN = 0.70   # a fuzzy similarity below this counts as no match (e.g. "headache" vs "stomach ache")


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def _windows(words, n):
    """All runs of n consecutive words (whole text if shorter)."""
    if len(words) <= n:
        return [" ".join(words)]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def _alias_score(t, words, alias):
    if t == alias or re.search(rf"\b{re.escape(alias)}\b", t):
        return 1.0
    n = len(alias.split())
    best = max(difflib.SequenceMatcher(None, w, alias).ratio() for w in _windows(words, n) + [t])
    return best if best >= FUZZY_MIN else 0.0


def match(text):
    """-> {sign: score in [0, 1]} for all 6 signs.
    Letters: the text is a single letter ("a", "letter b"). Actions: exact synonym or synonym as whole
    words inside the text = 1.0; otherwise the best typo-tolerant similarity to a synonym, either to the
    whole text or to a run of words of the synonym's length ("i have fevr" -> "fever")."""
    t = normalize(text)
    scores = {s: 0.0 for s in SIGNS}
    if not t:
        return scores
    m = re.fullmatch(r"(?:letter )?([abc])(?: letter)?", t)
    if m:
        scores[m.group(1).upper()] = 1.0
        return scores
    words = t.split()
    for action, aliases in ALIASES.items():
        scores[action] = max(_alias_score(t, words, a) for a in aliases)
    return scores


def in_vocabulary(scores):
    return max(scores.values()) >= config.RETRY_BELOW


def match_action(text):
    """Backward-compatible: action scores only."""
    s = match(text)
    return {a: s[a] for a in config.ACTIONS}


def supported_list():
    return ["A", "B", "C"] + [DISPLAY[a] for a in config.ACTIONS]


def reference_media(cls):
    """Path of the reference image (letter) or clip (action), or None."""
    if cls in config.LETTERS:
        cands = [config.ISL_ASSETS_DIR / f"{cls}{e}" for e in (".png", ".jpg", ".jpeg")]
    else:
        cands = [config.ISL_ASSETS_DIR / "actions" / f"{cls}{e}" for e in (".mp4", ".webm", ".mov")]
    return next((p for p in cands if p.exists()), None)
