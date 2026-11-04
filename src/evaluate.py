"""Held-out accuracy: letters, actions (split by signer) and handwriting.

Run:  python -m src.evaluate   (after extract / extract_actions / train)
Output: reports/accuracy.md, reports/accuracy.json, reports/actions_*.png
Test samples are never compared against themselves, and calibration never sees test data.
Each section runs only if its data exists.
"""
import json
import sys
from collections import Counter

import numpy as np

from src import charts, config, handwriting
from src.comparator import Comparator
from src.decision import decide

LETTERS = config.LETTERS
ACTION_CLASSES = config.ACTION_CLASSES


def score(preds, labels):
    """preds: list of (true, result dict). -> metrics dict."""
    n = len(preds)
    cm = np.zeros((len(labels), len(labels)), int)
    bands, final_ok, confirm_hit = Counter(), 0, 0
    per_class_bands = {l: Counter() for l in labels}
    for true, res in preds:
        d = decide(res["confidences"], res["out_of_range"])
        cm[labels.index(true), labels.index(d.top)] += 1
        bands[d.band] += 1
        per_class_bands[true][d.band] += 1
        if d.band == "final" and d.top == true:
            final_ok += 1
        if d.band == "confirm" and true in (d.ranked[0][0], d.ranked[1][0]):
            confirm_hit += 1
    per_class = {l: (cm[i, i] / cm[i].sum() if cm[i].sum() else None) for i, l in enumerate(labels)}
    return {
        "n": n,
        "top1_accuracy": float(np.trace(cm) / n),
        "per_class_accuracy": per_class,
        "per_class_n": {l: int(cm[i].sum()) for i, l in enumerate(labels)},
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "bands": {b: bands[b] for b in ("retry", "confirm", "final")},
        "per_class_bands": {l: {b: c[b] for b in ("retry", "confirm", "final")}
                            for l, c in per_class_bands.items()},
        "final_band_precision": final_ok / bands["final"] if bands["final"] else None,
        "confirm_band_top2_hit": confirm_hit / bands["confirm"] if bands["confirm"] else None,
    }


def top_confusions(r, n=3):
    cm, labels = np.array(r["confusion_matrix"]), r["labels"]
    pairs = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
             for j in range(len(labels)) if i != j and cm[i, j] > 0]
    return [{"true": t, "predicted": p, "count": int(c)} for c, t, p in sorted(pairs, reverse=True)[:n]]


# ---------- letters ----------
def eval_letters():
    if not config.SIGN_MODEL_PATH.exists():
        return None
    m = np.load(config.SIGN_MODEL_PATH)
    tr, te = m["train_idx"], m["test_idx"]
    comp = Comparator(m["X"][tr], m["y"][tr], float(m["T"]), float(m["d_max"]), int(m["k"]))
    preds = [(m["y"][i], comp.predict(np.stack([m["X"][i], m["Xm"][i]]))) for i in te]
    out = score(preds, LETTERS)
    out.update(train_size=int(len(tr)), T=float(m["T"]))
    return out


# ---------- actions ----------
def eval_actions(library_path=None):
    from src.dtw import ActionComparator, calibrate, library_dist
    from src.extract_actions import ACTIONS_LIBRARY_PATH
    path = library_path or ACTIONS_LIBRARY_PATH
    if not path.exists():
        return None
    lib = np.load(path)
    S, Sm, y, signer = lib["S"], lib["Sm"], lib["y"], lib["signer"]

    # Holistic detection rate over ALL clips (including unusable ones)
    sc, usable = lib["stat_cls"], lib["stat_usable"]
    detection = {c: {"clips": int((sc == c).sum()), "usable": int(usable[sc == c].sum()),
                     "body": float(lib["stat_pose"][sc == c].mean()),
                     "hands": float(lib["stat_hand"][sc == c].mean())}
                 for c in ACTION_CLASSES if (sc == c).any()}
    detection_all = {"clips": int(len(sc)), "usable": int(usable.sum()),
                     "body": float(lib["stat_pose"].mean()), "hands": float(lib["stat_hand"].mean())}
    out = {"detection": detection, "detection_all": detection_all,
           "signers": sorted(set(signer.tolist()))}
    if len(out["signers"]) < 2:
        out["error"] = "Need clips from at least 2 signers for a held-out split by signer."
        return out

    dist = library_dist(S, Sm)   # (N, N): row = query clip (min over its mirror), col = library clip
    preds, folds = [], {}
    for s in out["signers"]:
        tr, te = np.where(signer != s)[0], np.where(signer == s)[0]
        # calibration on the training signers only (leave-one-signer-out inside them)
        T, d_none = calibrate(dist[np.ix_(tr, tr)], y[tr], signer[tr])
        comp = ActionComparator(S[tr], y[tr], T, d_none)
        fold = [(y[i], comp.predict_from_dist(dist[i, tr])) for i in te]
        preds += fold
        fs = score(fold, ACTION_CLASSES)
        folds[s] = {"n": fs["n"], "top1_accuracy": fs["top1_accuracy"], "T": T, "d_none": d_none}
    out.update(score(preds, ACTION_CLASSES))
    cm = np.array(out["confusion_matrix"])
    ni = ACTION_CLASSES.index("NONE")
    none_n, act_n = cm[ni].sum(), cm[:ni].sum()
    out["folds"] = folds
    out["none_false_accept"] = float((none_n - cm[ni, ni]) / none_n) if none_n else None
    out["none_false_reject"] = float(cm[:ni, ni].sum() / act_n) if act_n else None
    out["top_confusions"] = top_confusions(out)
    return out


def action_charts(r, reports_dir):
    files = {}
    if r.get("detection"):
        p = reports_dir / "actions_detection.png"
        charts.detection(r["detection"], p, "Holistic detection rate per class (share of frames)")
        files["detection"] = p.name
    if "confusion_matrix" in r:
        p = reports_dir / "actions_confusion.png"
        charts.confusion(r["confusion_matrix"], r["labels"], p, "Actions: confusion matrix (held-out signers)")
        files["confusion"] = p.name
        p = reports_dir / "actions_per_class.png"
        charts.per_class(r["per_class_accuracy"], r["per_class_n"], p, "Actions: per-class accuracy")
        files["per_class"] = p.name
        p = reports_dir / "actions_bands.png"
        charts.bands(r["per_class_bands"], p, "Actions: decision band by true class")
        files["bands"] = p.name
    return files


# ---------- handwriting ----------
def eval_handwriting():
    d = config.HANDWRITING_TEST_DIR
    paths = config.iter_images(d) if d.exists() else []
    if not paths:
        return None
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
    out = score(preds, LETTERS)
    out["skipped"] = unreadable
    return out


# ---------- markdown ----------
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def metrics_md(r):
    labels = r["labels"]
    lines = [f"- Test samples: **{r['n']}**" + (f" (train library: {r['train_size']})" if "train_size" in r else ""),
             f"- **Top-1 accuracy: {pct(r['top1_accuracy'])}**",
             "- Per class: " + ", ".join(f"{l} {pct(a)}" for l, a in r["per_class_accuracy"].items()),
             f"- Decision bands: retry {r['bands']['retry']}, confirm {r['bands']['confirm']}, "
             f"final {r['bands']['final']}",
             f"- Precision when shown as final (>=85%): {pct(r['final_band_precision'])}",
             f"- Truth in top-2 when asking to confirm: {pct(r['confirm_band_top2_hit'])}",
             "", "Confusion matrix (rows = true, cols = predicted):", "",
             "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    lines += [f"| **{l}** | " + " | ".join(str(v) for v in row) + " |"
              for l, row in zip(labels, r["confusion_matrix"])]
    return lines + [""]


def actions_md(r, files):
    lines = ["## Actions -> Text (held-out by signer)", ""]
    if r is None:
        return lines + ["_No action library yet. Record clips, then run `python -m src.extract_actions`._", ""]
    da = r["detection_all"]
    lines += ["### Holistic detection rate", "",
              f"{da['usable']}/{da['clips']} clips usable; body visible in {pct(da['body'])} and a hand in "
              f"{pct(da['hands'])} of frames overall.", "",
              "| Class | Clips | Usable | Body | Hands |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {d['clips']} | {d['usable']} | {pct(d['body'])} | {pct(d['hands'])} |"
              for c, d in r["detection"].items()]
    lines += [""]
    if "detection" in files:
        lines += [f"![Detection rate]({files['detection']})", ""]
    if "error" in r:
        return lines + [f"**{r['error']}** Signers found: {', '.join(r['signers'])}.", ""]
    lines += ["### Held-out accuracy (leave-one-signer-out)", "",
              f"Signers: {', '.join(r['signers'])}. Each signer is tested against the other signers' clips; "
              "T and the NONE threshold are calibrated on the training signers only.", ""]
    lines += metrics_md(r)
    lines += [f"- NONE false accept (NONE clip shown as an action): {pct(r['none_false_accept'])}",
              f"- NONE false reject (action rejected as NONE): {pct(r['none_false_reject'])}",
              "- Per held-out signer: " + ", ".join(f"{s} {pct(f['top1_accuracy'])} (n={f['n']})"
                                                    for s, f in r["folds"].items()), ""]
    if r["top_confusions"]:
        lines += ["Most frequent confusions: " + "; ".join(
            f"{c['true']} -> {c['predicted']} ({c['count']})" for c in r["top_confusions"]), ""]
    for key, alt in (("confusion", "Confusion matrix"), ("per_class", "Per-class accuracy"),
                     ("bands", "Decision bands")):
        if key in files:
            lines += [f"![{alt}]({files[key]})", ""]
    return lines


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"signs": eval_letters(), "actions": eval_actions(), "handwriting": eval_handwriting()}
    files = action_charts(report["actions"], config.REPORTS_DIR) if report["actions"] else {}
    config.REPORT_JSON.write_text(json.dumps(report, indent=2))
    md = ["# SignBridge held-out accuracy", "",
          "Scope: 6 ISL signs only (letters A, B, C; actions BLOOD_PRESSURE, FEVER, STOMACH_ACHE). "
          "Not a general ISL translator.", "", "## Letters -> Text (held-out split, calibrated on train only)", ""]
    md += metrics_md(report["signs"]) if report["signs"] else ["_No letter model yet (needs data/raw)._", ""]
    md += actions_md(report["actions"], files)
    md += ["## Handwriting -> Sign (data/handwriting_test)", ""]
    md += metrics_md(report["handwriting"]) if report["handwriting"] else \
        ["_No test set yet (add images to data/handwriting_test/<A|B|C>/)._", ""]
    config.REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
