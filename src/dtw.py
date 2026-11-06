"""DTW distance + action comparator with a NONE ("not a known sign") class.

Distance to every library clip (min over original / mirrored query); per class the distance to the nearest
segments (augmented copies of one segment count once). Confidence combines the ranking (softmax) with
the absolute distance (a curve fitted on training distances); the remainder is NONE. T and the curve are
calibrated on training data only. The library is balanced per source video (select_library).
"""
import numpy as np

from src import config
from src.comparator import softmax_conf

CLASSES = config.ACTION_CLASSES   # (..actions.., "NONE")
BAND = 0.25                       # Sakoe-Chiba band as a fraction of sequence length
PALM_REL_SHOULDERS = (4, 18)      # columns of (x, y) palm position relative to the shoulder midpoint, per hand
MIN_MOTION_FRACTION = 0.5         # reject inputs that move less than half the least-moving training sign
AUG_MARK = "~aug-"
MAX_SEGMENTS_PER_SOURCE = 3       # library cap per source video
KEEP_AUG = ("w1", "w2")           # augmented copies kept per segment (time windows; mirrors are duplicates)


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


# ---------- library balancing ----------
def aug_kind(file):
    stem = str(file).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.split(AUG_MARK, 1)[1] if AUG_MARK in stem else None


def select_library(files, aug, parent, source):
    """Boolean mask of the clips that go into the matching library, so no source video / class dominates:
    - at most MAX_SEGMENTS_PER_SOURCE original segments per source video (evenly spread over the video);
    - per kept segment only the augmented kinds in KEEP_AUG (time windows). Mirrored copies are dropped:
      the comparator already matches every query against its own mirror image, so a mirrored library
      copy is an exact duplicate (verified: DTW 0.1-0.4 to the mirrored original vs 1.4-2.9 unmirrored)."""
    files, aug, parent, source = map(np.asarray, (files, aug, parent, source))
    keep_parent = set()
    for src in np.unique(source):
        origs = sorted(parent[(source == src) & ~aug])
        if len(origs) > MAX_SEGMENTS_PER_SOURCE:
            origs = [origs[i] for i in np.linspace(0, len(origs) - 1, MAX_SEGMENTS_PER_SOURCE).round().astype(int)]
        keep_parent.update(origs)
    kinds = [aug_kind(f) for f in files]
    return np.array([p in keep_parent and (not a or k in KEEP_AUG) for p, a, k in zip(parent, aug, kinds)])


# ---------- per-class distance ----------
def class_dists(d, y, parent, k=config.ACTION_K_NEAREST):
    """d: (N,) distances to library clips -> (len(CLASSES),) per-class distance = mean of the k nearest
    SEGMENTS of that class. A segment and its augmented copies count once (their minimum distance), so
    the k neighbours are k different segments, not k copies of one."""
    out = np.full(len(CLASSES), np.inf)
    for ci, c in enumerate(CLASSES):
        m = y == c
        if not m.any():
            continue
        best = {}
        for di, p in zip(d[m], parent[m]):
            if di < best.get(p, np.inf):
                best[p] = di
        if best:   # all clips of the class excluded (e.g. same signer during calibration) -> stays inf
            out[ci] = np.sort(list(best.values()))[:k].mean()
    return out


FIT_MAX = 0.97   # never fully certain: the distance curve is fitted on a handful of clips


def distance_fit(d, params):
    """P(the best match is correct | its distance d): monotone decreasing curve fitted on training clips."""
    if not np.isfinite(d):
        return 0.0
    return float(np.interp(d, params["fit_x"], params["fit_y"]))


def confidences(D, params):
    """Per-class distances -> confidences that use BOTH ranking and absolute distance.
    rel_c = softmax(-D_c / T) over the classes (ranking);  fit = P(best match correct | its distance),
    fitted on training clips (how close a correct match typically is). Action confidence = fit * rel_c;
    NONE ("Not a known sign") gets 1 - fit plus any ranking mass of real NONE clips. So a best match
    that is far compared with the usual within-class distances gets low confidence even if it ranks first."""
    D = np.asarray(D, np.float64)
    rel = softmax_conf(D, params["T"])
    fit = distance_fit(np.min(D[:-1]), params)
    conf = np.zeros(len(CLASSES))
    conf[:-1] = fit * rel[:-1]
    conf[-1] = 1.0 - conf[:-1].sum()
    return conf


def calibrate(dist, y, groups, parent, query_mask=None, k=config.ACTION_K_NEAREST):
    """Fit T and the distance curve on TRAINING clips only. Each query clip is scored only against clips
    from OTHER groups (signers), mimicking an unseen signer. Queries = original clips (query_mask).
      T:      min NLL of the ranking softmax for the true class.
      curve:  isotonic (monotone decreasing) regression of "best match is the true class" on the best
              match's distance, over all training queries. Queries whose class has no clip from another
              signer (or NONE clips) count as wrong at their best distance - like an unknown sign."""
    from sklearn.isotonic import IsotonicRegression
    dist = np.array(dist, np.float64)
    groups = np.asarray(groups)
    dist[groups[:, None] == groups[None, :]] = np.inf
    rows = np.where(query_mask)[0] if query_mask is not None else np.arange(len(y))
    raw = np.stack([class_dists(dist[i], y, parent, k) for i in rows])
    true_idx = np.array([CLASSES.index(c) for c in y[rows]])
    is_action = true_idx < len(CLASSES) - 1
    true_d = raw[np.arange(len(rows)), true_idx]
    ok = is_action & np.isfinite(true_d)
    if not ok.any():
        raise ValueError("No action clip has a same-class clip from another signer.")

    best_T, best_nll = 1.0, np.inf
    for T in np.logspace(-3, 1, 120):
        P = np.stack([softmax_conf(r, T) for r in raw[ok]])
        nll = -np.mean(np.log(P[np.arange(ok.sum()), true_idx[ok]] + 1e-12))
        if nll < best_nll:
            best_T, best_nll = float(T), nll

    d_best = raw[:, :-1].min(1)
    correct = (raw[:, :-1].argmin(1) == true_idx) & is_action
    m = np.isfinite(d_best)
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=FIT_MAX, out_of_bounds="clip")
    iso.fit(d_best[m], correct[m].astype(float))
    xs = np.unique(np.concatenate([d_best[m], [0.0, d_best[m].max() * 1.5]]))
    ys = iso.predict(xs)
    ys[xs > d_best[m].max()] = 0.0    # farther than any training query ever was -> not a known sign
    half = xs[ys < 0.5]
    return {"T": best_T, "fit_x": xs, "fit_y": ys, "n_correct": int(correct[m].sum()), "n_queries": int(m.sum()),
            "d_typical": float(np.median(true_d[ok])), "d_half": float(half.min()) if len(half) else float(xs[-1])}


class ActionComparator:
    def __init__(self, S, y, parent, params, k=config.ACTION_K_NEAREST, min_motion=0.0):
        self.S, self.y, self.parent = np.asarray(S, np.float32), np.asarray(y), np.asarray(parent)
        self.params, self.k, self.min_motion = dict(params), int(k), float(min_motion)

    def predict_from_dist(self, d, motion=None):
        D = class_dists(d, self.y, self.parent, self.k)
        conf = confidences(D, self.params)
        still = motion is not None and motion < self.min_motion   # an action sign needs hand movement
        far = bool(np.argmax(conf) == len(CLASSES) - 1)
        return {"confidences": dict(zip(CLASSES, conf.tolist())),
                "distances": dict(zip(CLASSES, D.tolist())),
                "dist_all": np.asarray(d),
                "motion": motion,
                "reject_reason": ("no hand movement" if still else
                                  "too far from every sign in the library" if far else None),
                "out_of_range": still}

    def predict(self, seq, seq_m):
        d = np.minimum(pairwise_dtw(seq[None], self.S), pairwise_dtw(seq_m[None], self.S))[0]
        return self.predict_from_dist(d, motion_extent(seq))
