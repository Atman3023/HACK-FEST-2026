"""Reference media for the app: the best teammate segment per action -> assets/isl/actions/<ACTION>.mp4
(browser-playable H.264), and one photo per letter from data/raw -> assets/isl/<LETTER>.jpg.

Usage (from the repo root):
  python scripts/make_reference_clips.py

Only teammate recordings are used for display (never the YouTube sources). Choices were made by
looking at reports/segments/*.png:
  BLOOD_PRESSURE  p2_1  complete sign from rest: hand to the upper arm (cuff), then the pumping hand
                        (p1_1 starts mid-sign with the hand already on the arm)
  FEVER           p1_2  full forehead -> under-chin movement starting from a neutral position
                        (p1_1 starts with the hand already on the forehead)
  STOMACH_ACHE    -     no teammate clip yet: the app shows a "Reference clip to be recorded" card.

H.264 is written through Windows Media Foundation (OpenCV MSMF backend), so no ffmpeg is needed.
Browsers cannot play OpenCV's default mp4v output.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

CHOICES = {"BLOOD_PRESSURE": "p2_1", "FEVER": "p1_2", "STOMACH_ACHE": None}
OUT_DIR = config.ISL_ASSETS_DIR / "actions"


def read(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


DISPLAY_WIDTH = 360   # MSMF has no bitrate control and writes ~6 MB/s at full size; shrink for the browser


def write_h264(path, frames, fps):
    if fps > 45:                           # 60 fps phone clips -> 30 fps
        frames, fps = frames[::2], fps / 2
    h, w = frames[0].shape[:2]
    s = DISPLAY_WIDTH / w
    frames = [cv2.resize(f, (DISPLAY_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA) for f in frames]
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2          # H.264 needs even dimensions
    vw = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), round(fps), (w2, h2))
    if not vw.isOpened():
        raise RuntimeError("Could not open an H.264 writer (Windows Media Foundation).")
    for f in frames:
        vw.write(f[:h2, :w2])
    vw.release()
    cap = cv2.VideoCapture(str(path))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return codec, n


def letter_images():
    """assets/isl/<L>.jpg = first photo of each letter in data/raw (resized to 640 px wide)."""
    for letter in config.LETTERS:
        d = config.RAW_DIR / letter
        imgs = config.iter_images(d) if d.exists() else []
        if not imgs:
            print(f"  {letter}: no photo in data/raw/{letter}/ - no reference image")
            continue
        img = cv2.imdecode(np.fromfile(str(imgs[0]), np.uint8), cv2.IMREAD_COLOR)
        s = 640 / img.shape[1]
        img = cv2.resize(img, (640, int(img.shape[0] * s)))
        out = config.ISL_ASSETS_DIR / f"{letter}.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out))
        print(f"  {letter}: {imgs[0].relative_to(ROOT)} -> {out.relative_to(ROOT)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    letter_images()
    missing = []
    for action, stem in CHOICES.items():
        out = OUT_DIR / f"{action}.mp4"
        src = config.ACTIONS_DIR / action / f"{stem}.mp4" if stem else None
        if src is None or not src.exists():
            missing.append(action)
            print(f"  {action}: no teammate clip -> the app shows a 'Reference clip to be recorded' card")
            continue
        frames, fps = read(src)
        codec, n = write_h264(out, frames, fps)
        print(f"  {action}: {src.relative_to(ROOT)} -> {out.relative_to(ROOT)} ({n} frames, {codec}, "
              f"{out.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"\nStill to record: {', '.join(missing)} (save as assets/isl/actions/<ACTION>.mp4, H.264).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
