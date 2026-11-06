"""Split long source videos into single-sign action clips using hand-motion activity.

Usage (from the repo root):
  python scripts/segment_videos.py            # all sources in SOURCES
  python scripts/segment_videos.py --no-aug   # segments only

Input:  data/sources/<CLASS>__<signer>.mp4  (ASCII-named copies; the originals are never touched)
Output:
  data/actions/<CLASS>/<signer>_<n>.mp4                 one original segment per sign repetition
  data/actions/<CLASS>/<signer>_<n>~aug-w1|w2|w3.mp4    overlapping time windows of that segment
  data/actions/<CLASS>/<signer>_<n>~aug-mirror.mp4      horizontally mirrored segment
  data/actions/segments_manifest.csv                    every clip: source, times, augmented flag
  reports/segments/<CLASS>_<signer>_<n>.png             contact sheet per segment
  reports/segments/timeline_<CLASS>__<signer>.png       activity signal + chosen segments per source

Augmented copies carry "~aug-" in the filename. src/extract_actions.py reads that marker so evaluation
only ever TESTS on original segments (augmented copies are training-only, and leave-one-signer-out
already keeps every copy of a test clip out of its training fold).

Activity per frame (MediaPipe Holistic, model_complexity=1):
  raised  = a pose wrist (15/16) is above the waist line
            (hip midpoint if visible, else shoulder midpoint + 1.3 x shoulder width)
  moving  = smoothed wrist speed (shoulder widths / s) with hysteresis: starts > MOVE_HIGH,
            continues while > MOVE_LOW
  active  = raised and moving; gaps < MERGE_GAP s bridged; spans > MAX_LEN s (several repetitions
            signed back to back) split at the slowest moment; pieces < MIN_LEN s dropped; PAD s padding.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")   # quieter MediaPipe logs
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

SOURCES_DIR = ROOT / "data" / "sources"
OUT_DIR = config.ACTIONS_DIR
REPORT_DIR = config.REPORTS_DIR / "segments"
MANIFEST = OUT_DIR / "segments_manifest.csv"
AUG_MARK = "~aug-"

# Per-source settings, chosen after looking at reports/segments/timeline_*.png:
#   trim     (start_s, end_s) of the source to use (rest is ignored)
#   single   the clip holds exactly ONE sign: keep first..last movement (speed > MOVE_LOW) as one
#            segment, never split
#   split_at extra cut times (s); each snaps to the slowest moment within +-0.25 s
#   drop     times (s); the segment containing that time is discarded (e.g. an "I" point before FEVER)
SOURCES = {
    "BLOOD_PRESSURE__p1": {"single": True},
    "BLOOD_PRESSURE__p2": {"single": True},
    "FEVER__p1": {"split_at": (1.9,)},   # forehead -> chin, twice
    # "I" point -> forehead -> chin, three times (the 2nd time forehead->chin is signed twice), plus
    # a hand drop at ~2.95 s and a hand raise at ~9.0 s.
    "FEVER__yt_fever": {"split_at": (0.8, 2.75, 4.7, 6.25, 9.25), "drop": (0.4, 2.95, 4.05, 9.0)},
    "STOMACH_ACHE__yt_sdlj": {"trim": (0.0, 10.5)},   # after 10.5 s: subscribe screen
}

MOVE_HIGH = 1.0      # shoulder widths / s: a movement starts above this ...
MOVE_LOW = 0.5       # ... and continues while speed stays above this (hysteresis)
SMOOTH_S = 0.15      # s; median + moving-average window on wrist positions / speed
WAIST_FROM_SHOULDERS = 1.5   # shoulder widths below the shoulder line, used when the hips are not visible
MERGE_GAP = 0.80     # s; bridge pauses shorter than this (holds inside one sign)
MIN_LEN = 0.80       # s; shorter pieces are dropped
MAX_LEN = 2.60       # s; longer spans hold several repetitions -> split at the deepest speed valley
PAD = 0.15           # s added at both ends of a segment
WINDOWS = ((0.0, 0.85), (0.15, 1.0), (0.075, 0.925))   # overlapping windows as fractions of the segment


# ---------- video io ----------
def read_frames(path, trim=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if trim:
        a, b = int(round(trim[0] * fps)), int(round(trim[1] * fps))
        frames = frames[a:b]
    return frames, fps


def write_clip(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


# ---------- activity ----------
def frame_signals(frames):
    """-> wrists (T, 2, 2) in shoulder-width units (nan if not visible), raised (T,)

    Uses the pose wrists (15/16): unlike hand landmarks they never swap source mid-clip,
    so the trajectory has no jumps."""
    import mediapipe as mp
    T = len(frames)
    wrists = np.full((T, 2, 2), np.nan)
    raised = np.zeros(T, bool)
    with mp.solutions.holistic.Holistic(model_complexity=1, static_image_mode=False) as hol:
        for t, f in enumerate(frames):
            h, w = f.shape[:2]
            r = hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks is None:
                continue
            P = r.pose_landmarks.landmark
            xy = lambda i: np.array([P[i].x * w, P[i].y * h])  # noqa: E731  pixels: x and y on one scale
            ls, rs = xy(11), xy(12)
            width = np.linalg.norm(ls - rs)
            if width < 1:
                continue
            if min(P[23].visibility, P[24].visibility) > 0.5:
                waist = (xy(23)[1] + xy(24)[1]) / 2
            else:   # hips out of frame (seated / close-up): estimate from the shoulders
                waist = (ls[1] + rs[1]) / 2 + WAIST_FROM_SHOULDERS * width
            for k, i in enumerate((15, 16)):
                if P[i].visibility > 0.3:
                    p = xy(i)
                    wrists[t, k] = p / width
                    raised[t] |= p[1] < waist
    return wrists, raised


def _smooth(x, n, kind):
    if n <= 1:
        return x
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return np.median(win, axis=1) if kind == "median" else win.mean(axis=1)


def activity(wrists, raised, fps):
    """-> speed (T,) [shoulder widths / s] of the faster raised wrist, active mask (T,)"""
    n = max(1, int(round(SMOOTH_S * fps)))
    T = len(wrists)
    speed = np.zeros(T)
    for k in range(2):
        pos = wrists[:, k].copy()
        for c in range(2):   # fill gaps, then median (kills single-frame glitches) + mean
            v = pos[:, c]; ok = ~np.isnan(v)
            if ok.sum() < 2:
                pos[:, c] = 0.0; continue
            v = np.interp(np.arange(T), np.where(ok)[0], v[ok])
            pos[:, c] = _smooth(_smooth(v, n, "median"), n, "mean")
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
        speed = np.maximum(speed, np.concatenate([[0.0], d]))
    speed = _smooth(speed, n, "mean")
    speed = np.where(raised, speed, 0.0)
    active = np.zeros(T, bool)
    on = False
    for t in range(T):   # hysteresis: start above MOVE_HIGH, keep going while above MOVE_LOW
        on = speed[t] > (MOVE_LOW if on else MOVE_HIGH)
        active[t] = on
    return speed, active


def _split(a, b, speed, fps):
    """Recursively split [a, b) at the slowest interior moment while it is longer than MAX_LEN."""
    if (b - a) / fps <= MAX_LEN:
        return [(a, b)]
    m = int(round(MIN_LEN / 2 * fps))
    lo, hi = a + m, b - m
    if hi <= lo:
        return [(a, b)]
    c = lo + int(np.argmin(speed[lo:hi]))
    return _split(a, c, speed, fps) + _split(c, b, speed, fps)


def to_segments(active, speed, fps, offset=0.0, opts=None):
    """-> list of (start_frame, end_frame, status) with status 'kept' | 'dropped' | 'short'."""
    opts = opts or {}
    runs, start = [], None
    for t, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = t
        elif not a and start is not None:
            runs.append([start, t]); start = None
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) / fps < MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    n, w = len(speed), int(round(0.15 * fps))
    if opts.get("single"):
        pieces = [(merged[0][0], merged[-1][1])] if merged else []
    else:
        cuts = []
        for t in opts.get("split_at", ()):
            c = int(round((t - offset) * fps))
            lo, hi = max(0, c - w), min(n, c + w + 1)
            if lo < hi:
                cuts.append(lo + int(np.argmin(speed[lo:hi])))
        pieces = []
        for a, b in merged:
            edges = [a] + sorted(c for c in cuts if a < c < b) + [b]
            for x, y in zip(edges[:-1], edges[1:]):
                pieces += _split(x, y, speed, fps)
    # Pad only natural movement boundaries; a cut between two back-to-back movements gets no padding,
    # otherwise each piece would pick up the end/start of its neighbour.
    natural = {e for r in merged for e in r}
    pad, out = int(round(PAD * fps)), []
    for a, b in pieces:
        drop = any(a <= (d - offset) * fps < b for d in opts.get("drop", ()))
        status = "dropped" if drop else "short" if (b - a) / fps < MIN_LEN else "kept"
        out.append((max(0, a - pad) if a in natural else a, min(n, b + pad) if b in natural else b, status))
    return out


# ---------- reports ----------
def contact_sheet(frames, title, path, n=8, thumb_h=240):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    thumbs = []
    for i in idx:
        f = frames[i]
        s = thumb_h / f.shape[0]
        t = cv2.resize(f, (int(f.shape[1] * s), thumb_h))
        cv2.putText(t, f"{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, f"{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        thumbs.append(t)
    grid = np.vstack([np.hstack(thumbs[:4]), np.hstack(thumbs[4:])])
    head = np.full((36, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), np.vstack([head, grid]))


def timeline(speed, raised, segs, fps, offset, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = offset + np.arange(len(speed)) / fps
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(t, speed, lw=1, color="#1f5fa8", label="wrist speed")
    ax.axhline(MOVE_HIGH, color="#888", ls="--", lw=0.8, label="start threshold")
    ax.axhline(MOVE_LOW, color="#bbb", ls=":", lw=0.8, label="keep threshold")
    ax.fill_between(t, 0, np.where(raised, speed.max() * 1.05 + 1e-6, 0), color="#e0e0e0", step="mid",
                    label="hand above waist", zorder=0)
    colors = {"kept": "#3a9a5b", "dropped": "#c0504d", "short": "#e0a030"}
    for a, b, status in segs:
        ax.axvspan(offset + a / fps, offset + b / fps, color=colors[status], alpha=0.25)
    ax.set_xlabel("seconds in source"); ax.set_ylabel("shoulder widths / s")
    ax.set_title(f"{title}  (green = kept, red = dropped by config, orange = too short)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------- main ----------
def clear_previous(cls, signer):
    for p in (OUT_DIR / cls).glob(f"{signer}_*.mp4"):
        if p.stem.split(AUG_MARK)[0].rsplit("_", 1)[0] == signer:
            p.unlink()
    for p in REPORT_DIR.glob(f"{cls}_{signer}_*.png"):
        p.unlink()


def process(name, opts, augment):
    cls, signer = name.split("__")
    trim = opts.get("trim")
    frames, fps = read_frames(SOURCES_DIR / f"{name}.mp4", trim)
    offset = trim[0] if trim else 0.0
    wrists, raised = frame_signals(frames)
    speed, active = activity(wrists, raised, fps)
    if opts.get("single"):   # one known sign: slower parts (e.g. the BP "pump") count as movement too
        active = speed > MOVE_LOW
    segs = to_segments(active, speed, fps, offset, opts)
    timeline(speed, raised, segs, fps, offset, f"{cls} / {signer}", REPORT_DIR / f"timeline_{name}.png")

    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    clear_previous(cls, signer)
    rows, n = [], 0
    for a, b, status in segs:
        t0, t1 = offset + a / fps, offset + b / fps
        print(f"  {name}: {t0:5.2f}-{t1:5.2f}s ({(b - a) / fps:.2f}s) {status}")
        if status != "kept":
            continue
        n += 1
        seg = frames[a:b]
        stem = f"{signer}_{n}"
        write_clip(OUT_DIR / cls / f"{stem}.mp4", seg, fps)
        contact_sheet(seg, f"{cls} / {stem}  {t0:.2f}-{t1:.2f}s of {name}.mp4",
                      REPORT_DIR / f"{cls}_{stem}.png")
        base = dict(cls=cls, signer=signer, source=f"{name}.mp4", parent=stem)
        rows.append({**base, "file": f"{cls}/{stem}.mp4", "start_s": round(t0, 3), "end_s": round(t1, 3),
                     "augmented": 0, "aug": ""})
        if not augment:
            continue
        L = len(seg)
        for k, (fa, fb) in enumerate(WINDOWS, 1):
            wa, wb = int(round(fa * L)), int(round(fb * L))
            f = f"{stem}{AUG_MARK}w{k}.mp4"
            write_clip(OUT_DIR / cls / f, seg[wa:wb], fps)
            rows.append({**base, "file": f"{cls}/{f}", "start_s": round(t0 + wa / fps, 3),
                         "end_s": round(t0 + wb / fps, 3), "augmented": 1, "aug": f"window{k}"})
        f = f"{stem}{AUG_MARK}mirror.mp4"
        write_clip(OUT_DIR / cls / f, [cv2.flip(x, 1) for x in seg], fps)
        rows.append({**base, "file": f"{cls}/{f}", "start_s": round(t0, 3), "end_s": round(t1, 3),
                     "augmented": 1, "aug": "mirror"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-aug", action="store_true", help="write original segments only")
    ap.add_argument("--only", help="process one source, e.g. FEVER__yt_fever")
    args = ap.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    old = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as fh:
            old = list(csv.DictReader(fh))
    rows = [r for r in old if args.only and r["source"] != f"{args.only}.mp4"]
    for name, opts in SOURCES.items():
        if args.only and name != args.only:
            continue
        if not (SOURCES_DIR / f"{name}.mp4").exists():
            print(f"missing source: {name}.mp4 - skipped"); continue
        rows += process(name, opts, not args.no_aug)

    fields = ["file", "cls", "signer", "source", "start_s", "end_s", "augmented", "aug", "parent"]
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print(f"\n{'class':<16}{'signer':<12}{'segments':>9}{'augmented':>11}")
    for key in sorted({(r["cls"], r["signer"]) for r in rows}):
        rs = [r for r in rows if (r["cls"], r["signer"]) == key]
        print(f"{key[0]:<16}{key[1]:<12}{sum(int(r['augmented']) == 0 for r in rs):>9}"
              f"{sum(int(r['augmented']) == 1 for r in rs):>11}")
    print(f"Manifest -> {MANIFEST}\nContact sheets -> {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
