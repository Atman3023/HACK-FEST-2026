"""Shared constants: vocabulary, thresholds, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Vocabulary is strictly these letters. Never claim general ISL translation.
LETTERS = ("A", "B", "C")

# Action signs (dynamic, body-relative). NONE = "not a known sign" rejection class.
ACTIONS = ("BLOOD_PRESSURE", "FEVER", "STOMACH_ACHE")
ACTION_CLASSES = ACTIONS + ("NONE",)
CLIP_SECONDS = 3.0
COUNTDOWN_SECONDS = 3

# Decision thresholds (top confidence)
RETRY_BELOW = 0.60   # < 60%  -> ask user to retry
FINAL_AT = 0.85      # >= 85% -> final answer; in between -> confirm top 2

# Comparator
K_NEAREST = 3                 # average the k nearest library samples per letter
ACTION_K_NEAREST = 1          # actions: nearest SEGMENT per class (copies of a segment count once). With 1-2
                              # segments per class per signer, averaging 3 mixes in other signers' versions.
HAND_COUNT_WEIGHT = 3.0       # feature that keeps 1-hand and 2-hand signs apart
OUT_OF_RANGE_FACTOR = 1.5     # best distance > factor * p99(train true-class distance) -> "not A/B/C"

# Held-out split
TEST_SIZE = 0.25
SEED = 42

# Paths
RAW_DIR = ROOT / "data" / "raw"
LIBRARY_PATH = ROOT / "data" / "library" / "landmarks.npz"
HANDWRITING_TEST_DIR = ROOT / "data" / "handwriting_test"
ACTIONS_DIR = ROOT / "data" / "actions"          # <CLASS>/<signer>_<n>.mp4
MODELS_DIR = ROOT / "models"
SIGN_MODEL_PATH = MODELS_DIR / "sign_model.npz"
HANDWRITING_MODEL_PATH = MODELS_DIR / "handwriting_model.npz"
ISL_ASSETS_DIR = ROOT / "assets" / "isl"
REPORTS_DIR = ROOT / "reports"
REPORT_JSON = REPORTS_DIR / "accuracy.json"
REPORT_MD = REPORTS_DIR / "accuracy.md"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_from_path(path: Path, root: Path):
    """Letter label for an image: nearest parent folder named A/B/C, else a filename
    starting with the letter followed by a non-letter (e.g. 'A_01.jpg', 'B (3).png')."""
    for parent in path.relative_to(root).parents:
        if parent.name.upper() in LETTERS:
            return parent.name.upper()
    stem = path.stem
    if stem and stem[0].upper() in LETTERS and (len(stem) == 1 or not stem[1].isalpha()):
        return stem[0].upper()
    return None


def iter_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
