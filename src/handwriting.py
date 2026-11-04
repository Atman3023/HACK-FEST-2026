"""Handwritten-letter recogniser for the drawing canvas (A, B, C only).

Templates are rendered from several fonts x stroke thicknesses (upper + lower case),
preprocessed exactly like a drawing, and matched with the shared Comparator.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.comparator import Comparator, calibrate

SIZE = 32
FONT_DIR = Path("C:/Windows/Fonts")
FONTS = ["arial.ttf", "arialbd.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf",
         "comic.ttf", "calibri.ttf", "tahoma.ttf", "trebuc.ttf", "segoeui.ttf", "consola.ttf",
         "lucon.ttf", "segoepr.ttf", "segoesc.ttf", "Inkfree.ttf"]
STROKES = (0, 3, 6, 10)


def preprocess(ink: np.ndarray):
    """ink: bool mask (True = pen). Returns a unit-norm feature vector, or None if empty."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 10:
        return None
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    side = int(max(h, w) * 1.15) + 2
    sq = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = crop
    img = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    v = img.ravel() - img.mean()
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-6 else None


def ink_from_rgba(rgba: np.ndarray):
    """Canvas image_data (H, W, 4): ink = visible and dark."""
    rgba = np.asarray(rgba)
    gray = rgba[..., :3].astype(np.float32).mean(axis=2)
    visible = rgba[..., 3] > 0 if rgba.shape[-1] == 4 else np.ones(gray.shape, bool)
    return visible & (gray < 128)


def ink_from_file(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img < 128
    if img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return ink_from_rgba(rgba)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128


def render_templates():
    X, y, groups = [], [], []
    for font_name in FONTS:
        path = FONT_DIR / font_name
        if not path.exists():
            continue
        font = ImageFont.truetype(str(path), 180)
        for letter in config.LETTERS:
            for ch in (letter, letter.lower()):
                for s in STROKES:
                    im = Image.new("L", (300, 300), 255)
                    ImageDraw.Draw(im).text((150, 150), ch, font=font, fill=0, anchor="mm",
                                            stroke_width=s, stroke_fill=0)
                    v = preprocess(np.asarray(im) < 128)
                    if v is not None:
                        X.append(v); y.append(letter); groups.append(font_name)
    if not X:
        raise RuntimeError(f"No template fonts found in {FONT_DIR}")
    return np.stack(X), np.array(y), np.array(groups)


def build_model():
    """Render templates and calibrate T leave-one-font-out (templates = training data)."""
    X, y, groups = render_templates()
    T, d_max = calibrate(X, y, groups=groups)
    return {"X": X, "y": y, "groups": groups, "T": T, "d_max": d_max}


def load_comparator(path=config.HANDWRITING_MODEL_PATH):
    if Path(path).exists():
        m = np.load(path, allow_pickle=False)
        X, y, T, d_max = m["X"], m["y"], float(m["T"]), float(m["d_max"])
    else:
        m = build_model()
        X, y, T, d_max = m["X"], m["y"], m["T"], m["d_max"]
    return Comparator(X, y, T, d_max)
