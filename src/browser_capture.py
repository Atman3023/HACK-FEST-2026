"""3-second action clips from the visitor's BROWSER camera (works on Streamlit Community Cloud).

A small custom component (src/components/clip_recorder/index.html, no build step) shows a mirrored live
preview and a 3-2-1 countdown, then captures JPEG frames for `seconds` and returns them to Python.
The local OpenCV recorder (src/capture.py) remains available as a fallback when running locally.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import streamlit.components.v1 as components

from src import config

_recorder = components.declare_component(
    "clip_recorder", path=str(Path(__file__).parent / "components" / "clip_recorder"))


def clip_recorder(key="clip_recorder", fps=15, width=480):
    """Renders the recorder. -> component value ({id, frames, duration}) of the last recording, or None."""
    return _recorder(seconds=config.CLIP_SECONDS, countdown=config.COUNTDOWN_SECONDS, fps=fps, width=width,
                     key=key, default=None)


def decode(value):
    """Component value -> (frames_bgr, fps)."""
    frames = []
    for url in value.get("frames", []):
        raw = base64.b64decode(url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    fps = len(frames) / max(float(value.get("duration") or config.CLIP_SECONDS), 1e-3)
    return frames, fps
