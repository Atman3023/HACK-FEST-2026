"""MediaPipe hand detection + landmark normalisation; builds the sign library from data/raw.

Run:  python -m src.extract
Output: data/library/landmarks.npz  (X, Xm = mirrored, y, files, n_hands)
"""
import sys
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

from src import config

_mp_hands = mp.solutions.hands
N_POINTS = 21
FEATURE_DIM = 2 * N_POINTS * 3 + 1   # two hands x 21 x (x, y, z) + hand-count feature


def read_image(path):
    """cv2.imread breaks on non-ASCII paths (OneDrive); decode from bytes instead."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


class HandDetector:
    def __init__(self):
        self._hands = _mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                      min_detection_confidence=0.5)

    def detect(self, bgr):
        """-> (list of (21, 3) arrays in pixel units, raw mediapipe result)."""
        h, w = bgr.shape[:2]
        res = self._hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        hands = [np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], np.float32)
                 for lm in (res.multi_hand_landmarks or [])]
        return hands, res

    def close(self):
        self._hands.close()


def normalize(hands):
    """Order hands left-to-right, centre on the wrist (or wrist midpoint for two hands),
    scale so the farthest point is at distance 1. Keeps the two hands' relative pose."""
    hands = sorted(hands, key=lambda p: p[0, 0])[:2]
    origin = np.mean([h[0] for h in hands], axis=0)
    pts = np.concatenate(hands) - origin
    scale = np.linalg.norm(pts, axis=1).max()
    pts = pts / (scale if scale > 1e-6 else 1.0)
    vec = np.zeros(FEATURE_DIM, np.float32)
    vec[:pts.size] = pts.ravel()
    vec[-1] = config.HAND_COUNT_WEIGHT * (len(hands) - 1)
    return vec


def features(hands):
    """-> (original, mirrored) vectors. Mirroring lets left- and right-handed signers match."""
    flip = np.array([-1.0, 1.0, 1.0], np.float32)
    return normalize(hands), normalize([h * flip for h in hands])


def draw_landmarks(bgr, res):
    out = bgr.copy()
    for lm in res.multi_hand_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(out, lm, _mp_hands.HAND_CONNECTIONS)
    return out


def main():
    paths = config.iter_images(config.RAW_DIR) if config.RAW_DIR.exists() else []
    if not paths:
        print(f"No images found in {config.RAW_DIR}. Expected data/raw/<A|B|C>/*.jpg")
        return 1

    det = HandDetector()
    X, Xm, y, files, n_hands = [], [], [], [], []
    seen, skipped_label, no_hand = Counter(), [], []
    for i, path in enumerate(paths, 1):
        label = config.label_from_path(path, config.RAW_DIR)
        if label is None:
            skipped_label.append(path)
            continue
        seen[label] += 1
        img = read_image(path)
        hands, _ = det.detect(img) if img is not None else ([], None)
        if not hands:
            no_hand.append(path)
            continue
        f, fm = features(hands)
        X.append(f); Xm.append(fm); y.append(label)
        files.append(str(path.relative_to(config.ROOT)).replace("\\", "/"))
        n_hands.append(min(len(hands), 2))
        if i % 50 == 0:
            print(f"  {i}/{len(paths)} images processed")
    det.close()

    if not X:
        print("MediaPipe found no hands in any labelled image.")
        return 1
    config.LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.LIBRARY_PATH, X=np.stack(X), Xm=np.stack(Xm), y=np.array(y),
             files=np.array(files), n_hands=np.array(n_hands))

    kept = Counter(y)
    print(f"\nSaved {len(X)} samples -> {config.LIBRARY_PATH.relative_to(config.ROOT)}")
    for letter in config.LETTERS:
        hc = Counter(n for n, l in zip(n_hands, y) if l == letter)
        print(f"  {letter}: {kept[letter]}/{seen[letter]} with hands detected "
              f"(1-hand: {hc[1]}, 2-hand: {hc[2]})")
    if skipped_label:
        print(f"  {len(skipped_label)} images skipped (no A/B/C label), e.g. {skipped_label[0]}")
    if no_hand:
        print(f"  {len(no_hand)} images skipped (no hand detected), e.g. {no_hand[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
