"""Build the deployed models from the extracted libraries. Calibration uses TRAINING data only.

Run:  python -m src.train   (after python -m src.extract and python -m src.extract_actions)
Output:
  models/sign_model.npz         letter library (all samples) + T / d_max + held-out split indices
  models/action_model.npz       action library (all clips incl. augmented copies) + T / d_none
  models/handwriting_model.npz  handwriting templates
"""
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from src import config, handwriting
from src.comparator import calibrate, pairwise_dist

ACTION_MODEL_PATH = config.MODELS_DIR / "action_model.npz"
MIN_PER_LETTER_FOR_SPLIT = 4   # below this a held-out letter split is impossible


def load_action_library(path=None):
    """Extracted library + per-clip metadata (older libraries: every clip is its own segment/source)."""
    from src.extract_actions import ACTIONS_LIBRARY_PATH
    lib = np.load(path or ACTIONS_LIBRARY_PATH)
    n = len(lib["y"])
    files = lib["files"]
    meta = {
        "aug": lib["aug"] if "aug" in lib.files else np.zeros(n, bool),
        "parent": lib["parent"] if "parent" in lib.files else np.array([f.split("~aug-")[0].rsplit(".", 1)[0] for f in files]),
        "source": lib["source"] if "source" in lib.files else files,
    }
    return lib, meta


def train_actions():
    """Deployed action model: balanced library (select_library); T, the distance curve and the motion gate
    are calibrated leave-one-signer-out on the library clips (training data only)."""
    from src.dtw import calibrate as dtw_calibrate, calibrate_min_motion, library_dist, select_library
    from src.extract_actions import ACTIONS_LIBRARY_PATH
    if not ACTIONS_LIBRARY_PATH.exists():
        print("No action library (python -m src.extract_actions) - skipping actions.")
        return False
    lib, meta = load_action_library()
    keep = select_library(lib["files"], meta["aug"], meta["parent"], meta["source"])
    S, Sm, y, signer, files = (lib[k][keep] for k in ("S", "Sm", "y", "signer", "files"))
    aug, parent, source = meta["aug"][keep], meta["parent"][keep], meta["source"][keep]
    if len(set(signer.tolist())) < 2:
        print("Action model needs clips from at least 2 signers - skipping actions.")
        return False
    params = dtw_calibrate(library_dist(S, Sm), y, signer, parent, query_mask=~aug)
    min_motion = calibrate_min_motion(S[~aug], y[~aug])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ACTION_MODEL_PATH, S=S, Sm=Sm, y=y, signer=signer, aug=aug, files=files, parent=parent,
             source=source, T=params["T"], fit_x=params["fit_x"], fit_y=params["fit_y"],
             d_typical=params["d_typical"],
             d_half=params["d_half"], min_motion=min_motion, k=config.ACTION_K_NEAREST,
             has_none=bool((y == "NONE").any()))
    print(f"Action library: {int(keep.sum())}/{len(keep)} clips kept after balancing "
          f"(<= 3 segments per source video, 2 time-window copies per segment, no mirror copies)")
    for c in config.ACTION_CLASSES:
        m = y == c
        if m.any():
            print(f"  {c:<15} {int((m & ~aug).sum())} segments + {int((m & aug).sum())} copies "
                  f"from {len(set(source[m]))} source video(s)")
    print(f"Calibration: T={params['T']:.4f}; typical same-class distance {params['d_typical']:.2f}, "
          f"distance factor drops below 50% at {params['d_half']:.2f}; min_motion={min_motion:.3f}")
    if not (y == "NONE").any():
        print("WARNING: no NONE clips. 'Not a known sign' then relies only on the distance curve and the "
              "motion gate (both calibrated on training clips).\n"
              "         Record non-sign movements (waving, scratching head, idle) into data/actions/NONE/ "
              "for a much more reliable rejection.")
    return True


def fallback_letter_calibration(X, y, Xm):
    """With < 2 samples per letter there are no same-letter pairs, so T / d_max cannot be fitted.
    Fall back to the geometry of the library: d_between = smallest distance between samples of two
    different letters. A query at distance 0 from one letter and d_between from the others gets ~95%
    (T = d_between / 3.6); anything farther than d_between from every sample is out of range."""
    d = np.minimum(pairwise_dist(X, X), pairwise_dist(Xm, X))
    diff = y[:, None] != y[None, :]
    d_between = float(d[diff].min()) if diff.any() else 1.0
    return d_between / 3.6, d_between


def train_letters():
    if not config.LIBRARY_PATH.exists():
        print("No letter library yet. Run: python -m src.extract")
        return False
    lib = np.load(config.LIBRARY_PATH)
    X, Xm, y = lib["X"], lib["Xm"], lib["y"]
    counts = Counter(y.tolist())
    print("Letter library:", {l: counts[l] for l in config.LETTERS})
    idx = np.arange(len(y))
    can_split = all(counts[l] >= MIN_PER_LETTER_FOR_SPLIT for l in config.LETTERS)
    if can_split:
        train_idx, test_idx = train_test_split(idx, test_size=config.TEST_SIZE,
                                               random_state=config.SEED, stratify=y)
        # Calibration sees training samples only; the test split is untouched until evaluate.py.
        T, d_max = calibrate(X[train_idx], y[train_idx], Xm=Xm[train_idx])
        calibrated = True
    else:
        train_idx, test_idx = idx, np.array([], int)
        if all(counts[l] >= 2 for l in config.LETTERS):
            T, d_max = calibrate(X, y, Xm=Xm)
            calibrated = True
        else:
            T, d_max = fallback_letter_calibration(X, y, Xm)
            calibrated = False
        print(f"WARNING: fewer than {MIN_PER_LETTER_FOR_SPLIT} images for some letters - the library uses "
              "every image and held-out letter accuracy cannot be measured."
              + ("" if calibrated else " T / d_max are NOT calibrated (one image per letter); "
                 "add more photos per letter to data/raw/<A|B|C>/."))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.SIGN_MODEL_PATH, X=X, Xm=Xm, y=y, files=lib["files"],
             train_idx=train_idx, test_idx=test_idx, T=T, d_max=d_max, k=config.K_NEAREST,
             calibrated=calibrated)
    print(f"Sign model: library={len(y)} train={len(train_idx)} test={len(test_idx)} "
          f"T={T:.4f} d_max={d_max:.3f}{'' if calibrated else ' (uncalibrated fallback)'}")
    return True


def train_handwriting():
    hw = handwriting.build_model()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.HANDWRITING_MODEL_PATH, **hw)
    print(f"Handwriting model: {len(hw['y'])} templates from {len(set(hw['groups']))} fonts, "
          f"T={hw['T']:.4f} d_max={hw['d_max']:.3f}")


def main():
    ok_actions = train_actions()
    ok_letters = train_letters()
    train_handwriting()
    return 0 if (ok_actions or ok_letters) else 1


if __name__ == "__main__":
    sys.exit(main())
