"""Record a short clip from the LOCAL webcam (OpenCV) with a live Streamlit preview + countdown.

The app runs on the same laptop as the camera, so OpenCV reads it directly.
"""
import time

import cv2

from src import config


def _overlay(frame, big=None, status=None):
    img = cv2.flip(frame, 1)  # selfie preview; the recorded frames are not mirrored
    h, w = img.shape[:2]
    if big:
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
        cv2.putText(img, big, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 5,
                    (0, 220, 255), 10, cv2.LINE_AA)
    if status:
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


BUSY_MSG = ("The webcam is busy (another app or the browser camera may still be using it). "
            "Close other camera apps, or reload the page, then press Record again.")


def _open_working_camera(placeholder, camera, attempts=4, warmup=1.0, retry_wait=1.0):
    """Open the webcam and make sure it delivers REAL frames. When another process (e.g. the browser
    camera of "Spell a letter") still holds the device, OpenCV 'opens' it but returns black frames.
    The browser releases the camera asynchronously after a mode switch, so retry a few times."""
    for attempt in range(attempts):
        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            good, t0 = 0, time.time()
            while time.time() - t0 < warmup:  # also lets exposure settle
                ok, frame = cap.read()
                if ok and frame is not None and frame.mean() > 8:  # busy device -> all-black frames
                    good += 1
                    placeholder.image(_overlay(frame, status="Get into position..."), use_column_width=True)
            if good >= 5:
                return cap
        cap.release()
        if attempt < attempts - 1:
            placeholder.info("Waiting for the webcam...")
            time.sleep(retry_wait)
    raise RuntimeError(BUSY_MSG)


def record_clip(placeholder, seconds=config.CLIP_SECONDS, countdown=config.COUNTDOWN_SECONDS, camera=0):
    """Shows live frames in `placeholder` (an st.empty()). -> (frames_bgr, fps) or raises RuntimeError."""
    cap = _open_working_camera(placeholder, camera)
    try:
        t0 = time.time()
        while (left := countdown - (time.time() - t0)) > 0:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("The webcam stopped sending frames.")
            placeholder.image(_overlay(frame, big=str(int(left) + 1)), use_column_width=True)
        frames, t0, n = [], time.time(), 0
        while (elapsed := time.time() - t0) < seconds:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("The webcam stopped sending frames.")
            frames.append(frame)
            n += 1
            if n % 2 == 0:  # updating every frame slows the capture down
                placeholder.image(_overlay(frame, status=f"REC  {seconds - elapsed:.1f}s"),
                                  use_column_width=True)
        if len(frames) < 5 * seconds:  # well under 5 fps means the camera stalled
            raise RuntimeError(BUSY_MSG)
        return frames, len(frames) / max(time.time() - t0, 1e-6)
    finally:
        cap.release()
