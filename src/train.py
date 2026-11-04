"""Split the sign library, calibrate on TRAINING data only, build handwriting templates.

Run:  python -m src.train   (after python -m src.extract)
Output: models/sign_model.npz, models/handwriting_model.npz
"""
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from src import config, handwriting
from src.comparator import calibrate


ACTION_MODEL_PATH = config.MODELS_DIR / "action_model.npz"


def train_actions():
    """Deployed action model: library = all clips; T / d_none calibrated leave-one-signer-out on them."""
    from src.dtw import calibrate, library_dist
    from src.extract_actions import ACTIONS_LIBRARY_PATH
    if not ACTIONS_LIBRARY_PATH.exists():
        print("No action library (python -m src.extract_actions) - skipping actions.")
        return
    lib = np.load(ACTIONS_LIBRARY_PATH)
    S, Sm, y, signer = lib["S"], lib["Sm"], lib["y"], lib["signer"]
    if len(set(signer.tolist())) < 2:
        print("Action model needs clips from at least 2 signers - skipping actions.")
        return
    T, d_none = calibrate(library_dist(S, Sm), y, signer)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ACTION_MODEL_PATH, S=S, Sm=Sm, y=y, signer=signer, T=T, d_none=d_none, k=config.K_NEAREST)
    print(f"Action model: {len(y)} clips {dict(Counter(y.tolist()))}, T={T:.4f} d_none={d_none:.3f}")


def main():
    train_actions()
    if not config.LIBRARY_PATH.exists():
        print("No sign library yet. Run: python -m src.extract")
        return 1
    lib = np.load(config.LIBRARY_PATH)
    X, Xm, y = lib["X"], lib["Xm"], lib["y"]
    counts = Counter(y.tolist())
    print("Library:", dict(counts))
    missing = [l for l in config.LETTERS if counts[l] < 4]
    if missing:
        print(f"Need at least 4 samples per letter for a held-out split; too few for {missing}.")
        return 1

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=config.TEST_SIZE,
                                           random_state=config.SEED, stratify=y)
    # Calibration sees training samples only; the test split is untouched until evaluate.py.
    T, d_max = calibrate(X[train_idx], y[train_idx], Xm=Xm[train_idx])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.SIGN_MODEL_PATH, X=X, Xm=Xm, y=y, files=lib["files"],
             train_idx=train_idx, test_idx=test_idx, T=T, d_max=d_max, k=config.K_NEAREST)
    print(f"Sign model: train={len(train_idx)} test={len(test_idx)} T={T:.4f} d_max={d_max:.3f}")

    hw = handwriting.build_model()
    np.savez(config.HANDWRITING_MODEL_PATH, **hw)
    print(f"Handwriting model: {len(hw['y'])} templates from {len(set(hw['groups']))} fonts, "
          f"T={hw['T']:.4f} d_max={hw['d_max']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
