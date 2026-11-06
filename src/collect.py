"""In-app data collection: save a recorded clip into the action library folders and count clips.

Clips are saved like scripts/record_actions.py does: data/actions/<CLASS>/<signer>_<n>.mp4, not mirrored,
n continuing after the signer's existing clips (augmented "~aug-" copies are ignored).
"""
import io
import re
from collections import Counter
from contextlib import redirect_stdout

import cv2

from src import config

AUG_MARK = "~aug-"


def clean_signer(name):
    return re.sub(r"[^a-z0-9-]", "", name.strip().lower())


def next_index(cls, signer):
    pat = re.compile(re.escape(signer) + r"_(\d+)")
    d = config.ACTIONS_DIR / cls
    nums = [int(m.group(1)) for p in d.glob("*.mp4") if (m := pat.fullmatch(p.stem))] if d.exists() else []
    return max(nums, default=0) + 1


def save_clip(frames, fps, cls, signer):
    """-> saved path. Raises ValueError for an empty signer name or unknown class."""
    signer = clean_signer(signer)
    if not signer:
        raise ValueError("Enter a signer name (letters, digits or '-').")
    if cls not in config.ACTION_CLASSES:
        raise ValueError(f"Unknown class {cls}")
    path = config.ACTIONS_DIR / cls / f"{signer}_{next_index(cls, signer)}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def clip_counts():
    """-> {class: Counter(signer -> original clips)} (augmented copies not counted)."""
    out = {}
    for cls in config.ACTION_CLASSES:
        c = Counter()
        d = config.ACTIONS_DIR / cls
        for p in (d.glob("*.mp4") if d.exists() else []):
            if AUG_MARK not in p.stem:
                c[p.stem.rsplit("_", 1)[0]] += 1
        out[cls] = c
    return out


def rebuild_library():
    """Feature extraction (cached: only new/changed clips run MediaPipe) + action library build.
    -> (ok, log text)."""
    from src import extract_actions, train
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = extract_actions.main()
        ok = rc == 0 and train.train_actions()
    return bool(ok), buf.getvalue()
