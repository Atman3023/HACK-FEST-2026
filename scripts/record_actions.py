"""Record 3-second action clips for the SignBridge action library.

Usage:
  python scripts/record_actions.py                            # asks for class + signer
  python scripts/record_actions.py --class FEVER --signer atman

Keys (click the video window first):
  Enter / Space  record the next clip (3-2-1 countdown, then 3 s)
  1-4            switch class
  D              delete the last clip saved in this session
  Q / Esc        quit

Clips are saved to data/actions/<CLASS>/<signer>_<n>.mp4 (n continues after existing clips).
Preview is mirrored like a selfie; the saved clip is NOT mirrored (same as the app's capture).
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

WINDOW = "SignBridge action recorder"
CLASSES = config.ACTION_CLASSES
SHORT = {"BLOOD_PRESSURE": "BP", "FEVER": "FEVER", "STOMACH_ACHE": "STOMACH", "NONE": "NONE"}
WHITE, BLACK, RED, GREEN, YELLOW = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 200, 0), (0, 220, 255)


# ---------- files ----------
def clean_signer(name):
    return re.sub(r"[^a-z0-9-]", "", name.strip().lower())


def next_index(class_dir: Path, signer: str):
    pat = re.compile(re.escape(signer) + r"_(\d+)")
    nums = [int(m.group(1)) for p in class_dir.glob("*.mp4") if (m := pat.fullmatch(p.stem))]
    return max(nums, default=0) + 1


def count_clips(out_dir: Path):
    """-> {class: Counter(signer -> n clips)}"""
    out = {}
    for cls in CLASSES:
        c = Counter()
        for p in (out_dir / cls).glob("*.mp4"):
            if "~aug-" not in p.stem:   # augmented copies from scripts/segment_videos.py
                c[p.stem.rsplit("_", 1)[0]] += 1
        out[cls] = c
    return out


def save_clip(frames, fps, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    check = cv2.VideoCapture(str(path))
    n = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    return n


def print_table(counts, signer):
    signers = sorted({s for c in counts.values() for s in c} | {signer})
    width = max(8, *(len(s) for s in signers))
    print("\n" + "signer".ljust(width) + "".join(SHORT[c].rjust(9) for c in CLASSES))
    for s in signers:
        mark = " <" if s == signer else ""
        print(s.ljust(width) + "".join(str(counts[c][s]).rjust(9) for c in CLASSES) + mark)
    print("TOTAL".ljust(width) + "".join(str(sum(counts[c].values())).rjust(9) for c in CLASSES))


# ---------- drawing ----------
def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), BLACK, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(frame, cls, signer, counts, status, status_color=WHITE, big=None):
    img = cv2.flip(frame, 1)  # selfie preview
    h, w = img.shape[:2]
    text(img, f"Class: {cls}   Signer: {signer}", (10, 25), 0.65, YELLOW, 2)
    text(img, "This signer: " + "  ".join(f"{SHORT[c]} {counts[c][signer]}" for c in CLASSES), (10, 52))
    text(img, "All signers: " + "  ".join(f"{SHORT[c]} {sum(counts[c].values())}" for c in CLASSES), (10, 76))
    text(img, status, (10, h - 15), 0.65, status_color, 2)
    if big is not None:
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
        cv2.putText(img, big, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, YELLOW, 10, cv2.LINE_AA)
    return img


# ---------- capture ----------
def countdown_and_record(cap, cls, signer, counts):
    """3-2-1 countdown with live preview, then record CLIP_SECONDS by wall clock.
    Returns (frames, fps) or None if cancelled / camera failed."""
    t0 = time.time()
    while (left := config.COUNTDOWN_SECONDS - (time.time() - t0)) > 0:
        ok, frame = cap.read()
        if not ok:
            return None
        cv2.imshow(WINDOW, overlay(frame, cls, signer, counts, "Get ready...  (Esc = cancel)",
                                   big=str(int(left) + 1)))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            return None
    frames, t0 = [], time.time()
    while (elapsed := time.time() - t0) < config.CLIP_SECONDS:
        ok, frame = cap.read()
        if not ok:
            return None
        frames.append(frame)
        img = overlay(frame, cls, signer, counts, f"REC  {config.CLIP_SECONDS - elapsed:.1f}s", RED)
        cv2.circle(img, (img.shape[1] - 30, 30), 12, RED, -1)
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)
    elapsed = time.time() - t0
    return (frames, len(frames) / elapsed) if frames else None


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return None
    t0 = time.time()
    while time.time() - t0 < 1.5:  # first frames are dark while auto-exposure settles
        cap.read()
    return cap


class QualityChecker:
    """Runs MediaPipe Holistic over a saved clip: how often are body and hands visible?"""

    def __init__(self):
        import mediapipe as mp
        self._holistic = mp.solutions.holistic.Holistic(model_complexity=1)

    def check(self, frames, need_hands=True):
        pose = hands = 0
        for f in frames:
            r = self._holistic.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            pose += r.pose_landmarks is not None
            hands += (r.left_hand_landmarks is not None) or (r.right_hand_landmarks is not None)
        n = len(frames)
        ok = pose >= 0.8 * n and (hands >= 0.3 * n or not need_hands)  # NONE may be idle, no hands
        return ok, f"body {100 * pose // n}%, hands {100 * hands // n}% of frames"

    def close(self):
        self._holistic.close()


def ask_class():
    print("Classes:")
    for i, c in enumerate(CLASSES, 1):
        print(f"  {i}. {c}")
    while True:
        s = input("Class number or name: ").strip().upper()
        if s.isdigit() and 1 <= int(s) <= len(CLASSES):
            return CLASSES[int(s) - 1]
        if s in CLASSES:
            return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", choices=CLASSES)
    ap.add_argument("--signer")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.ACTIONS_DIR)
    ap.add_argument("--auto", type=int, default=0, help="record N clips without waiting for keys (testing)")
    ap.add_argument("--no-check", action="store_true", help="skip the Holistic quality check after each clip")
    args = ap.parse_args(argv)

    cls = args.cls or ask_class()
    signer = clean_signer(args.signer or input("Signer name (letters/digits/-): "))
    while not signer:
        signer = clean_signer(input("Signer name (letters/digits/-): "))

    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not open camera {args.camera}. Close other apps using the webcam and retry.")
        return 1

    checker = None if args.no_check else QualityChecker()
    counts = count_clips(args.out)
    print_table(counts, signer)
    print("\nClick the video window, then: Enter = record, 1-4 = class, D = delete last, Q/Esc = quit")
    saved, recorded_auto = [], 0
    msg, msg_color = "Enter = record   1-4 = class   D = delete last   Q = quit", WHITE
    try:
        while True:
            if args.auto and recorded_auto >= args.auto:
                break
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            cv2.imshow(WINDOW, overlay(frame, cls, signer, counts, msg, msg_color))
            key = 13 if args.auto else cv2.waitKey(15) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the mouse
            if key in (27, ord("q"), ord("Q")):
                break
            if ord("1") <= key < ord("1") + len(CLASSES):
                cls = CLASSES[key - ord("1")]
                msg, msg_color = f"Switched to {cls}", YELLOW
            elif key in (ord("d"), ord("D")) and saved:
                last = saved.pop()
                last.unlink(missing_ok=True)
                counts = count_clips(args.out)
                msg, msg_color = f"Deleted {last.parent.name}/{last.name}", YELLOW
                print(f"Deleted {last}")
            elif key in (13, 10, 32):
                result = countdown_and_record(cap, cls, signer, counts)
                recorded_auto += 1
                if result is None:
                    msg, msg_color = "Cancelled", YELLOW
                    continue
                frames, fps = result
                path = args.out / cls / f"{signer}_{next_index(args.out / cls, signer)}.mp4"
                n = save_clip(frames, fps, path)
                if n <= 0:
                    msg, msg_color = f"FAILED to save {path.name}", RED
                    print(f"Failed to write {path}")
                    continue
                saved.append(path)
                counts = count_clips(args.out)
                msg, msg_color = f"Saved {cls}/{path.name}  ({n} frames, {fps:.0f} fps)", GREEN
                print(f"Saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
                      f"  ({n} frames @ {fps:.1f} fps)")
                if checker:
                    good, summary = checker.check(frames, need_hands=cls != "NONE")
                    if good:
                        msg += f"  | {summary}"
                        print(f"  quality OK: {summary}")
                    else:
                        msg, msg_color = f"Saved {path.name} but POOR: {summary}. D = delete", RED
                        print(f"  WARNING poor clip: {summary} - press D to delete and re-record")
                print_table(counts, signer)
    finally:
        if checker:
            checker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
