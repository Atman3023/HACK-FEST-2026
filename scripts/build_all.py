"""One command: extract features -> build libraries/models in models/ -> held-out evaluation.

Usage (from the repo root):
  python scripts/build_all.py

Steps
  1. letters  data/raw/<A|B|C>/*      -> data/library/landmarks.npz  (MediaPipe Hands)
  2. actions  data/actions/<CLASS>/*  -> data/library/actions.npz    (MediaPipe Holistic, 32-frame sequences)
  3. models   models/sign_model.npz, models/action_model.npz, models/handwriting_model.npz
              (each holds the full library the app matches against + calibration from training data only)
  4. report   reports/accuracy.md + .json + charts (leave-one-signer-out for actions)
A step that has no data is skipped with a message; later steps still run.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, evaluate, extract, extract_actions, train  # noqa: E402


def step(name, fn):
    print(f"\n=== {name} " + "=" * max(3, 60 - len(name)))
    t0 = time.time()
    rc = fn()
    print(f"--- {name}: {'ok' if rc in (0, True, None) else 'skipped / incomplete'} ({time.time() - t0:.0f}s)")
    return rc


def main():
    step("1/4 Letter features (data/raw)", extract.main)
    step("2/4 Action features (data/actions)", extract_actions.main)
    step("3/4 Models -> models/", train.main)
    step("4/4 Held-out evaluation -> reports/accuracy.md", evaluate.main)

    print("\n=== Summary " + "=" * 50)
    for p in (config.SIGN_MODEL_PATH, train.ACTION_MODEL_PATH, config.HANDWRITING_MODEL_PATH, config.REPORT_MD):
        print(f"  {'OK     ' if p.exists() else 'MISSING'} {p.relative_to(config.ROOT)}")
    if not (config.ACTIONS_DIR / "NONE").exists() or not any((config.ACTIONS_DIR / "NONE").glob("*.mp4")):
        print("\nWARNING: no NONE clips in data/actions/NONE/. Unknown movements are rejected only by the "
              "calibrated DTW distance threshold.\n         Recommended: record 5+ non-sign clips per signer "
              "(waving, scratching head, adjusting glasses, idle):\n"
              "           python scripts/record_actions.py --class NONE --signer <name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
