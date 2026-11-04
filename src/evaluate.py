"""Held-out accuracy for the sign comparator and the handwriting recogniser.

Run:  python -m src.evaluate   (after python -m src.train)
Output: reports/accuracy.md, reports/accuracy.json
Test samples are compared against TRAINING samples only, never against themselves.
"""
import json
import sys
from collections import Counter

import numpy as np

from src import config, handwriting
from src.comparator import Comparator
from src.decision import decide

L = config.LETTERS


def score(preds):
    """preds: list of (true, result dict). -> metrics dict."""
    n = len(preds)
    cm = np.zeros((len(L), len(L)), int)
    bands, final_ok, confirm_hit = Counter(), 0, 0
    for true, res in preds:
        d = decide(res["confidences"], res["out_of_range"])
        cm[L.index(true), L.index(d.top)] += 1
        bands[d.band] += 1
        if d.band == "final" and d.top == true:
            final_ok += 1
        if d.band == "confirm" and true in (d.ranked[0][0], d.ranked[1][0]):
            confirm_hit += 1
    per_letter = {l: (cm[i, i] / cm[i].sum() if cm[i].sum() else None) for i, l in enumerate(L)}
    return {
        "n": n,
        "top1_accuracy": float(np.trace(cm) / n),
        "per_letter_accuracy": per_letter,
        "confusion_matrix": cm.tolist(),
        "bands": {b: bands[b] for b in ("retry", "confirm", "final")},
        "final_band_precision": final_ok / bands["final"] if bands["final"] else None,
        "confirm_band_top2_hit": confirm_hit / bands["confirm"] if bands["confirm"] else None,
    }


def eval_signs():
    m = np.load(config.SIGN_MODEL_PATH)
    tr, te = m["train_idx"], m["test_idx"]
    comp = Comparator(m["X"][tr], m["y"][tr], float(m["T"]), float(m["d_max"]), int(m["k"]))
    preds = [(m["y"][i], comp.predict(np.stack([m["X"][i], m["Xm"][i]]))) for i in te]
    out = score(preds)
    out.update(train_size=int(len(tr)), T=float(m["T"]))
    return out


def eval_handwriting():
    d = config.HANDWRITING_TEST_DIR
    paths = config.iter_images(d) if d.exists() else []
    comp = handwriting.load_comparator()
    preds, unreadable = [], 0
    for p in paths:
        label = config.label_from_path(p, d)
        ink = handwriting.ink_from_file(p)
        v = handwriting.preprocess(ink) if ink is not None else None
        if label is None or v is None:
            unreadable += 1
            continue
        preds.append((label, comp.predict(v)))
    if not preds:
        return None
    out = score(preds)
    out["skipped"] = unreadable
    return out


def fmt_pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def section(title, r):
    lines = [f"## {title}", ""]
    if r is None:
        return lines + ["_No test set yet (add images to data/handwriting_test/<A|B|C>/)._", ""]
    lines += [f"- Test samples: **{r['n']}**" + (f" (train library: {r['train_size']})" if "train_size" in r else ""),
              f"- **Top-1 accuracy: {fmt_pct(r['top1_accuracy'])}**",
              "- Per letter: " + ", ".join(f"{l} {fmt_pct(a)}" for l, a in r["per_letter_accuracy"].items()),
              f"- Decision bands: retry {r['bands']['retry']}, confirm {r['bands']['confirm']}, final {r['bands']['final']}",
              f"- Precision when shown as final (>=85%): {fmt_pct(r['final_band_precision'])}",
              f"- Truth in top-2 when asking to confirm: {fmt_pct(r['confirm_band_top2_hit'])}",
              "", "Confusion matrix (rows = true, cols = predicted):", "",
              "| | " + " | ".join(L) + " |", "|---" * (len(L) + 1) + "|"]
    lines += [f"| **{l}** | " + " | ".join(str(v) for v in row) + " |"
              for l, row in zip(L, r["confusion_matrix"])]
    return lines + [""]


def main():
    if not config.SIGN_MODEL_PATH.exists():
        print("No model yet. Run: python -m src.extract && python -m src.train")
        return 1
    report = {"signs": eval_signs(), "handwriting": eval_handwriting()}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_JSON.write_text(json.dumps(report, indent=2))
    md = ["# SignBridge held-out accuracy", "",
          "Scope: ISL letters A, B and C only. Not a general ISL translator.", ""]
    md += section("Sign -> Text (held-out split, calibrated on train only)", report["signs"])
    md += section("Handwriting -> Sign (data/handwriting_test)", report["handwriting"])
    config.REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
