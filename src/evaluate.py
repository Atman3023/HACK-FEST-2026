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


def score(preds, labels, reject_label=None):
    """preds: list of (true, result dict). -> metrics dict.
    reject_label: class counted as the prediction when the input is rejected (out of range / NONE)."""
    n = len(preds)
    cm = np.zeros((len(labels), len(labels)), int)
    bands, final_ok, confirm_hit = Counter(), 0, 0
    per_class_bands = {l: Counter() for l in labels}
    for true, res in preds:
        d = decide(res["confidences"], res["out_of_range"])
        pred = reject_label if reject_label and (res["out_of_range"] or d.top == reject_label) else d.top
        cm[labels.index(true), labels.index(pred)] += 1
        bands[d.band] += 1
        per_class_bands[true][d.band] += 1
        if d.band == "final" and pred == true:
            final_ok += 1
        if d.band == "confirm" and true in (d.ranked[0][0], d.ranked[1][0]):
            confirm_hit += 1
    per_class = {l: (cm[i, i] / cm[i].sum() if cm[i].sum() else None) for i, l in enumerate(labels)}
    return {
        "n": n,
        "top1_accuracy": float(np.trace(cm) / n) if n else None,
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
    counts = {l: int((m["y"] == l).sum()) for l in LETTERS}
    if len(te) == 0:
        return {"error": "Held-out letter accuracy cannot be measured: the letter library has "
                         + ", ".join(f"{l}: {c}" for l, c in counts.items())
                         + " image(s); at least 4 per letter are needed for a held-out split.",
                "library": counts, "calibrated": bool(m["calibrated"]) if "calibrated" in m.files else True}
    comp = Comparator(m["X"][tr], m["y"][tr], float(m["T"]), float(m["d_max"]), int(m["k"]))
    preds = [(m["y"][i], comp.predict(np.stack([m["X"][i], m["Xm"][i]]))) for i in te]
    out = score(preds, LETTERS)
    out.update(train_size=int(len(tr)), T=float(m["T"]))
    return out


# ---------- actions ----------
def eval_actions(library_path=None):
    from src.dtw import (ActionComparator, calibrate, calibrate_min_motion, library_dist, motion_extent,
                         select_library)
    from src.extract_actions import ACTIONS_LIBRARY_PATH
    from src.train import load_action_library
    path = library_path or ACTIONS_LIBRARY_PATH
    if not path.exists():
        return None
    lib, meta = load_action_library(path)
    S, Sm, y, signer = lib["S"], lib["Sm"], lib["y"], lib["signer"]
    # augmented copies are training-only: never a test clip
    aug, parent = meta["aug"], meta["parent"]
    # training folds use the same balanced library as the app (caps per source video, no mirror copies);
    # every original clip of the held-out signer is tested
    in_lib = select_library(lib["files"], aug, parent, meta["source"])

    # Holistic detection rate over ALL clips (including unusable ones)
    sc, usable = lib["stat_cls"], lib["stat_usable"]
    detection = {c: {"clips": int((sc == c).sum()), "usable": int(usable[sc == c].sum()),
                     "body": float(lib["stat_pose"][sc == c].mean()),
                     "hands": float(lib["stat_hand"][sc == c].mean())}
                 for c in ACTION_CLASSES if (sc == c).any()}
    detection_all = {"clips": int(len(sc)), "usable": int(usable.sum()),
                     "body": float(lib["stat_pose"].mean()), "hands": float(lib["stat_hand"].mean())}
    out = {"detection": detection, "detection_all": detection_all,
           "signers": sorted(set(signer.tolist())), "augmented_train_only": int(aug.sum())}
    # Group = signer. Every source video belongs to exactly one signer, so holding a signer out keeps
    # every segment AND every augmented copy of the test videos out of the training fold.
    orig_signers = {c: sorted(set(signer[(y == c) & ~aug].tolist())) for c in ACTION_CLASSES}
    out["sources_per_class"] = {c: s for c, s in orig_signers.items() if s}
    out["measurable"] = [c for c, s in orig_signers.items() if len(s) >= 2]
    out["unmeasurable"] = [c for c, s in orig_signers.items() if len(s) == 1]
    out["has_none_clips"] = bool((y == "NONE").any())
    if len(out["signers"]) < 2:
        out["error"] = "Need clips from at least 2 signers for a held-out split by signer."
        return out

    dist = library_dist(S, Sm)   # (N, N): row = query clip (min over its mirror), col = library clip
    preds, folds = [], {}
    for s in out["signers"]:
        tr, te = np.where((signer != s) & in_lib)[0], np.where((signer == s) & ~aug)[0]
        if len(te) == 0:
            continue
        try:   # calibration on the training signers only (leave-one-signer-out inside them)
            params = calibrate(dist[np.ix_(tr, tr)], y[tr], signer[tr], parent[tr], query_mask=~aug[tr])
        except ValueError as e:
            folds[s] = {"n": int(len(te)), "skipped": str(e)}
            continue
        tro = tr[~aug[tr]]
        comp = ActionComparator(S[tr], y[tr], parent[tr], params, min_motion=calibrate_min_motion(S[tro], y[tro]))
        fold = [(y[i], comp.predict_from_dist(dist[i, tr], motion_extent(S[i]))) for i in te]
        preds += fold
        fs = score([p for p in fold if p[0] in out["measurable"]], ACTION_CLASSES, reject_label="NONE")
        folds[s] = {"n": fs["n"], "classes": sorted(set(y[te].tolist()) & set(out["measurable"])),
                    "top1_accuracy": fs["top1_accuracy"], "T": params["T"], "d_half": params["d_half"],
                    "calib_correct": params["n_correct"], "calib_queries": params["n_queries"]}
    # Headline numbers only over classes with >= 2 sources: a class with one source never has a
    # same-class training clip when its source is held out, so its "accuracy" would be meaningless.
    measured = [p for p in preds if p[0] in out["measurable"]]
    out.update(score(measured, ACTION_CLASSES, reject_label="NONE"))
    out["per_class_accuracy"] = {c: (v if c in out["measurable"] else None)
                                 for c, v in out["per_class_accuracy"].items()}
    out["unmeasurable_n"] = sum(p[0] in out["unmeasurable"] for p in preds)
    # ranking only: the closest action class, before the confidence / rejection decision
    acts = list(ACTION_CLASSES[:-1])
    ranked = [(t, acts[int(np.argmin([r["distances"][a] for a in acts]))]) for t, r in measured]
    out["ranked_per_class"] = {c: (float(np.mean([p == t for t, p in ranked if t == c]))
                                   if any(t == c for t, _ in ranked) else None) for c in out["measurable"]}
    out["ranked_top1"] = float(np.mean([p == t for t, p in ranked])) if ranked else None
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
    src = r["sources_per_class"]
    lines += ["### Held-out accuracy (leave-one-signer-out)", "",
              "Each signer is held out in turn and tested against the other signers' clips. Every source "
              "video belongs to one signer, so no segment or augmented copy of a test video is ever in "
              "training; augmented copies are training-only and never used as test clips. Training folds use "
              "the balanced library (at most 3 segments per source video, 2 time-window copies each, no mirror "
              "copies). Confidence = ranking x absolute-distance curve; T, the curve and the minimum-motion "
              "gate are calibrated on the training signers only.", "",
              "| Class | Sources (signers) | Held-out accuracy |", "|---|---|---|"]
    for c in ACTION_CLASSES:
        if c not in src:
            continue
        acc = (pct(r["per_class_accuracy"][c]) + f" (n={r['per_class_n'][c]})" if c in r["measurable"]
               else "**cannot be measured** - only one source")
        lines.append(f"| {c} | {len(src[c])}: {', '.join(src[c])} | {acc} |")
    lines.append("")
    for c in r["unmeasurable"]:
        lines += [f"> **{c}: held-out accuracy cannot be measured.** All its clips come from one source "
                  f"({src[c][0]}); when that source is held out, no {c} clip is left to match against. "
                  f"The clips are still in the app's library, but they are left out of the numbers below. "
                  "Record the sign by a second person to measure it.", ""]
    if not r["has_none_clips"]:
        lines += ['> **No NONE clips.** "Not a known sign" relies only on the calibrated distance curve '
                  '(a best match far from the typical same-class distance) and the motion gate. Record '
                  'non-sign movements into `data/actions/NONE/` to measure and improve rejection.', ""]
    if r["n"] == 0:
        return lines + ["_No measurable test clips._", ""]
    lines += [f"Numbers below cover the measurable classes only ({', '.join(r['measurable'])}). "
              "Confidence bands: retry < 60%, confirm top-2 60-85%, final >= 85%.", ""]
    lines += [f"- **Ranking only** (closest class, before confidence/rejection): {pct(r['ranked_top1'])} - "
              + ", ".join(f"{c} {pct(v)}" for c, v in r["ranked_per_class"].items()),
              "- **Decision** (what the app shows, with confidence bands and rejection):", ""]
    lines += metrics_md(r)
    weak = [s for s, f in r["folds"].items() if f.get("calib_correct", 99) < 3 and "skipped" not in f]
    if weak:
        lines += [f"> **Absolute-distance curve not estimable in fold(s) {', '.join(weak)}:** with that signer "
                  "held out, fewer than 3 training clips have a correct best match from another signer, so the "
                  "curve learns that no distance is trustworthy and the fold rejects its test clips. More "
                  "signers per class fix this.", ""]
    if r["has_none_clips"]:
        lines += [f"- NONE false accept (NONE clip shown as an action): {pct(r['none_false_accept'])}"]
    lines += [f"- Actions rejected as not a known sign: {pct(r['none_false_reject'])}",
              "- Per held-out signer (measurable classes only): " + ", ".join(
                  f"{s} skipped ({f['skipped']})" if "skipped" in f
                  else f"{s} not measurable (only one-source classes)" if f["n"] == 0
                  else f"{s} {pct(f['top1_accuracy'])} (n={f['n']}: {', '.join(f['classes'])}; "
                       f"calibration: {f['calib_correct']}/{f['calib_queries']} correct best matches)"
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
    if report["signs"] is None:
        md += ["_No letter model yet (needs data/raw)._", ""]
    elif "error" in report["signs"]:
        md += [f"**{report['signs']['error']}**", ""]
        if not report["signs"]["calibrated"]:
            md += ["The app still recognises letters from this library, but its confidence scale and "
                   "rejection distance are an uncalibrated fallback, not tuned on data.", ""]
    else:
        md += metrics_md(report["signs"])
    md += actions_md(report["actions"], files)
    md += ["## Handwriting -> Sign (data/handwriting_test)", ""]
    md += metrics_md(report["handwriting"]) if report["handwriting"] else \
        ["_No test set yet (add images to data/handwriting_test/<A|B|C>/)._", ""]
    config.REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
