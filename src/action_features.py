"""MediaPipe Holistic -> body-relative per-frame features -> fixed-length sequence.

Per hand (left, right), all positions divided by shoulder width:
  palm centre relative to nose, chin, shoulder midpoint and the OPPOSITE upper arm (x, y each = 8)
  hand shape: 5 fingertip-to-wrist distances / palm size                           (5)
  presence: 1 = hand landmarks, 0.5 = pose-wrist fallback, 0 = missing               (1)
=> 14 per hand, 28 per frame. Missing frames are interpolated; clips resampled to N_FRAMES.
"""
import cv2
import mediapipe as mp
import numpy as np

N_FRAMES = 32
PER_HAND = 14
DIM = 2 * PER_HAND
POS_X_COLS = [c + h * PER_HAND for h in (0, 1) for c in (0, 2, 4, 6)]  # x of the 4 relative positions
# Resting hand (below the frame), in shoulder widths: rel nose, chin, shoulder mid, opposite arm; shape; flag
REST = np.array([0, 3.5, 0, 3.0, 0, 2.5, 0, 2.0] + [1.0] * 5 + [0.0], np.float32)
WEIGHTS = np.array(([1.0] * 8 + [0.5] * 5 + [0.5]) * 2, np.float32)    # positions > shape/flag

# Pose landmark ids
NOSE, MOUTH_L, MOUTH_R, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 9, 10, 11, 12, 13, 14, 15, 16
CHIN = 152          # face mesh
TIPS, PALM = (4, 8, 12, 16, 20), (0, 5, 17)


def new_holistic():
    return mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)


def _xy(lms, w, h):
    return np.array([[p.x * w, p.y * h] for p in lms.landmark], np.float32)


def frame_features(res, w, h):
    """-> (28,) vector with NaN for missing parts, or None if no usable pose."""
    if res.pose_landmarks is None:
        return None
    pose = _xy(res.pose_landmarks, w, h)
    vis = np.array([p.visibility for p in res.pose_landmarks.landmark])
    sw = np.linalg.norm(pose[L_SH] - pose[R_SH])
    if sw < 1e-3 or vis[L_SH] < 0.3 or vis[R_SH] < 0.3:
        return None
    nose = pose[NOSE]
    if res.face_landmarks is not None:
        chin = _xy(res.face_landmarks, w, h)[CHIN]
    else:  # chin ~ mouth centre pushed away from the nose
        mouth = (pose[MOUTH_L] + pose[MOUTH_R]) / 2
        chin = mouth + (mouth - nose) * 1.2
    mid_sh = (pose[L_SH] + pose[R_SH]) / 2
    upper_arm = {"left": (pose[L_SH] + pose[L_EL]) / 2, "right": (pose[R_SH] + pose[R_EL]) / 2}

    out = np.full(DIM, np.nan, np.float32)
    for i, (side, hand_lms, wrist_id) in enumerate((("left", res.left_hand_landmarks, L_WR),
                                                    ("right", res.right_hand_landmarks, R_WR))):
        o = i * PER_HAND
        opposite = upper_arm["right" if side == "left" else "left"]
        if hand_lms is not None:
            hp = _xy(hand_lms, w, h)
            palm = hp[list(PALM)].mean(axis=0)
            size = np.linalg.norm(hp[9] - hp[0]) + 1e-6
            out[o + 8:o + 13] = np.linalg.norm(hp[list(TIPS)] - hp[0], axis=1) / size
            out[o + 13] = 1.0
        elif vis[wrist_id] > 0.5:
            palm = pose[wrist_id]
            out[o + 13] = 0.5
        else:
            out[o + 13] = 0.0
            continue
        rel = np.concatenate([palm - nose, palm - chin, palm - mid_sh, palm - opposite]) / sw
        out[o:o + 8] = rel
    return out


def to_sequence(frames_feats, min_valid=0.5):
    """List of per-frame vectors (or None) -> (N_FRAMES, DIM) weighted sequence, or None."""
    n = len(frames_feats)
    valid = [f is not None for f in frames_feats]
    if n == 0 or sum(valid) < max(2, min_valid * n):
        return None
    M = np.stack([f if f is not None else np.full(DIM, np.nan, np.float32) for f in frames_feats])
    t = np.arange(n)
    for c in range(DIM):
        ok = ~np.isnan(M[:, c])
        if ok.sum() == 0:
            # never-seen hand = resting below the frame (NOT 0, which would mean "palm on the nose")
            M[:, c] = REST[c % PER_HAND]
        elif ok.sum() < n:
            M[:, c] = np.interp(t, t[ok], M[ok, c])
    tt = np.linspace(0, n - 1, N_FRAMES)
    seq = np.stack([np.interp(tt, t, M[:, c]) for c in range(DIM)], axis=1)
    return (seq * WEIGHTS).astype(np.float32)


def mirror(seq):
    """Left/right mirror: swap hand blocks and flip the sign of x positions."""
    m = np.concatenate([seq[:, PER_HAND:], seq[:, :PER_HAND]], axis=1).copy()
    m[:, POS_X_COLS] *= -1
    return m


def draw(frame_bgr, res):
    """Pose skeleton + both hands drawn on a copy of the frame."""
    du, hol, styles = mp.solutions.drawing_utils, mp.solutions.holistic, mp.solutions.drawing_styles
    out = frame_bgr.copy()
    if res.pose_landmarks is not None:
        du.draw_landmarks(out, res.pose_landmarks, hol.POSE_CONNECTIONS,
                          landmark_drawing_spec=styles.get_default_pose_landmarks_style())
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand is not None:
            du.draw_landmarks(out, hand, hol.HAND_CONNECTIONS,
                              styles.get_default_hand_landmarks_style(),
                              styles.get_default_hand_connections_style())
    return out


def process_frames(frames_bgr, annotate=False):
    """-> (sequence or None, stats dict[, annotated frames]). One fresh Holistic per clip."""
    hol = new_holistic()
    feats, drawn, pose_n, hand_n = [], [], 0, 0
    try:
        for f in frames_bgr:
            h, w = f.shape[:2]
            res = hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            pose_n += res.pose_landmarks is not None
            hand_n += (res.left_hand_landmarks is not None) or (res.right_hand_landmarks is not None)
            feats.append(frame_features(res, w, h))
            if annotate:
                drawn.append(draw(f, res))
    finally:
        hol.close()
    n = max(len(frames_bgr), 1)
    stats = {"frames": len(frames_bgr), "pose_rate": pose_n / n, "hand_rate": hand_n / n}
    if annotate:
        return to_sequence(feats), stats, drawn
    return to_sequence(feats), stats


def to_gif(frames_bgr, fps, width=360, step=2, mirror_view=True):
    """Annotated frames -> animated GIF bytes (plays inline in st.image, no codec issues)."""
    import io
    from PIL import Image
    imgs = []
    for f in frames_bgr[::step]:
        f = cv2.flip(f, 1) if mirror_view else f
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(1000 * step / max(fps, 1)))
    return buf.getvalue()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames
