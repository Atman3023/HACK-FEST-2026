"""DTW distance + action comparator with a NONE ("not a known sign") class.

Distance to every library clip (min over original / mirrored query), mean of the k nearest per
class, softmax(-d / T). NONE's distance = min(mean k nearest NONE clips, d_none), so NONE wins
when every real action is too far away. T and d_none are calibrated on training data only.
"""
import numpy as np

from src import config
from src.comparator import letter_kmean, softmax_conf

CLASSES = config.ACTION_CLASSES   # (..actions.., "NONE")
BAND = 0.25                       # Sakoe-Chiba band as a fraction of sequence length
PALM_REL_SHOULDERS = (4, 18)      # columns of (x, y) palm position relative to the shoulder midpoint, per hand
MIN_MOTION_FRACTION = 0.5         # reject inputs that move less than half the least-moving training sign


def motion_extent(seq):
    """How far the hands travel in a clip: max over hands of the diagonal of the box the palm covers,
    in shoulder widths (body-relative, so swaying does not count). A still person scores ~0.05."""
    return float(max(np.linalg.norm(seq[:, c:c + 2].max(0) - seq[:, c:c + 2].min(0)) for c in PALM_REL_SHOULDERS))


def calibrate_min_motion(S, y):
    """Motion gate from TRAINING clips: a fraction of the smallest motion among real action clips."""
    m = [motion_extent(s) for s, c in zip(S, y) if c != "NONE"]
    return MIN_MOTION_FRACTION * min(m) if m else 0.0


def dtw_batch(A, B, band=BAND):
    """A, B: (P, T, D) -> (P,) DTW distances, path-length normalised. DP vectorised over pairs."""
    P, T, _ = A.shape
    cost = np.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1))  # (P, T, T)
    w = max(1, int(round(band * T)))
    acc = np.full((P, T + 1, T + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(max(1, i - w), min(T, i + w) + 1):
            acc[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(
                np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    return acc[:, T, T] / (2 * T)


def pairwise_dtw(Q, S, chunk=512):
    """Q: (m, T, D), S: (n, T, D) -> (m, n)."""
    m, n = len(Q), len(S)
    qi, si = np.repeat(np.arange(m), n), np.tile(np.arange(n), m)
    out = np.empty(m * n)
    for s in range(0, m * n, chunk):
        out[s:s + chunk] = dtw_batch(Q[qi[s:s + chunk]], S[si[s:s + chunk]])
    return out.reshape(m, n)


def library_dist(S, Sm):
    """All-pairs clip distance, min over mirroring the query."""
    return np.minimum(pairwise_dtw(S, S), pairwise_dtw(Sm, S))


def class_dists(d, y, d_none, k=config.K_NEAREST):
    """d: (N,) distances to library clips -> (len(CLASSES),) per-class distance."""
    D = letter_kmean(d, y, CLASSES, k)
    D[-1] = min(D[-1], d_none)   # NONE: real NONE clips or the "too far from everything" level
    return D


def calibrate(dist, y, groups, k=config.K_NEAREST):
    """Tune T and d_none (min NLL) on training clips; each clip is scored only against clips
    from OTHER groups (signers), so calibration mimics an unseen signer."""
    dist = np.array(dist, np.float64)
    groups = np.asarray(groups)
    dist[groups[:, None] == groups[None, :]] = np.inf
    raw = np.stack([letter_kmean(dist[i], y, CLASSES, k) for i in range(len(y))])
    true_idx = np.array([CLASSES.index(c) for c in y])
    action_rows = true_idx < len(CLASSES) - 1
    true_d = raw[np.arange(len(y)), true_idx][action_rows]
    true_d = true_d[np.isfinite(true_d)]
    if len(true_d) == 0:
        raise ValueError("No action clip has a same-class clip from another signer.")

    candidates = np.percentile(true_d, [50, 60, 70, 80, 90, 95, 99, 100])
    candidates = np.unique(np.concatenate([candidates, candidates * 1.25, candidates * 1.5]))
    has_none = (~action_rows).any()
    if not has_none:  # nothing to tune d_none against -> conservative fixed rule
        candidates = np.array([np.percentile(true_d, 99) * config.OUT_OF_RANGE_FACTOR])

    best = (np.inf, 1.0, candidates[-1])
    for d_none in candidates:
        D = raw.copy()
        D[:, -1] = np.minimum(D[:, -1], d_none)
        for T in np.logspace(-4, 1, 80):
            P = np.stack([softmax_conf(r, T) for r in D])
            nll = -np.mean(np.log(P[np.arange(len(D)), true_idx] + 1e-12))
            if nll < best[0]:
                best = (nll, float(T), float(d_none))
    return best[1], best[2]


class ActionComparator:
    def __init__(self, S, y, T, d_none, k=config.K_NEAREST, min_motion=0.0):
        self.S, self.y = np.asarray(S, np.float32), np.asarray(y)
        self.T, self.d_none, self.k = float(T), float(d_none), int(k)
        self.min_motion = float(min_motion)

    def predict_from_dist(self, d, motion=None):
        D = class_dists(d, self.y, self.d_none, self.k)
        conf = softmax_conf(D, self.T)
        too_far = bool(np.min(D[:-1]) > self.d_none)    # farther than the calibrated rejection distance
        still = motion is not None and motion < self.min_motion   # an action sign needs hand movement
        return {"confidences": dict(zip(CLASSES, conf.tolist())),
                "distances": dict(zip(CLASSES, D.tolist())),
                "motion": motion,
                "reject_reason": "no hand movement" if still else "too different from every sign" if too_far else None,
                "out_of_range": too_far or still}

    def predict(self, seq, seq_m):
        d = np.minimum(pairwise_dtw(seq[None], self.S), pairwise_dtw(seq_m[None], self.S))[0]
        return self.predict_from_dist(d, motion_extent(seq))
