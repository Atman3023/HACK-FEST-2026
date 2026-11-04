"""Brute-force nearest-sample comparator, shared by the sign and handwriting recognisers.

For a query, distance to EVERY library sample is computed; per letter the k nearest
distances are averaged; distances become confidences via softmax(-d / T).
T and the out-of-range distance are calibrated on TRAINING data only.
"""
import numpy as np

from src import config


def pairwise_dist(A, B):
    """Euclidean distances (len(A), len(B)) without a huge broadcast."""
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


def letter_kmean(d, y, letters, k):
    """d: (N,) distances to library samples -> (len(letters),) mean of k nearest per letter."""
    out = np.full(len(letters), np.inf)
    for i, letter in enumerate(letters):
        dl = d[y == letter]
        if dl.size:
            out[i] = np.sort(dl)[:k].mean()
    return out


def softmax_conf(D, T):
    s = -np.asarray(D, dtype=np.float64) / T
    s = s - np.max(s[np.isfinite(s)]) if np.isfinite(s).any() else np.zeros_like(s)
    e = np.where(np.isfinite(s), np.exp(s), 0.0)
    return e / e.sum() if e.sum() > 0 else np.full_like(e, 1.0 / len(e))


def calibrate(X, y, letters=config.LETTERS, k=config.K_NEAREST, Xm=None, groups=None):
    """Tune temperature T (min NLL) and out-of-range distance on the given (training) set.

    Each sample is scored against the others; samples sharing its group (itself by default,
    or e.g. the same font) are excluded so calibration never sees a sample matched to itself.
    Xm: optional mirrored variants (query distance = min over variants).
    """
    dist = pairwise_dist(X, X)
    if Xm is not None:
        dist = np.minimum(dist, pairwise_dist(Xm, X))
    groups = np.arange(len(X)) if groups is None else np.asarray(groups)
    dist[groups[:, None] == groups[None, :]] = np.inf

    D = np.stack([letter_kmean(dist[i], y, letters, k) for i in range(len(X))])
    true_idx = np.array([letters.index(l) for l in y])
    true_d = D[np.arange(len(X)), true_idx]
    ok = np.isfinite(true_d)
    D, true_idx, true_d = D[ok], true_idx[ok], true_d[ok]

    best_T, best_nll = 1.0, np.inf
    for T in np.logspace(-5, 1, 240):
        P = np.stack([softmax_conf(row, T) for row in D])
        nll = -np.mean(np.log(P[np.arange(len(D)), true_idx] + 1e-12))
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    d_max = float(np.percentile(true_d, 99) * config.OUT_OF_RANGE_FACTOR) if len(true_d) else np.inf
    return best_T, d_max


class Comparator:
    def __init__(self, X, y, T, d_max=np.inf, k=config.K_NEAREST, letters=config.LETTERS):
        self.X, self.y = np.asarray(X, np.float32), np.asarray(y)
        self.T, self.d_max, self.k, self.letters = float(T), float(d_max), int(k), tuple(letters)

    def predict(self, variants):
        """variants: (m, D) query versions (e.g. original + mirrored); per-sample distance = min."""
        variants = np.atleast_2d(np.asarray(variants, np.float32))
        d = pairwise_dist(variants, self.X).min(axis=0)
        D = letter_kmean(d, self.y, self.letters, self.k)
        conf = softmax_conf(D, self.T)
        return {
            "confidences": dict(zip(self.letters, conf.tolist())),
            "distances": dict(zip(self.letters, D.tolist())),
            "out_of_range": bool(np.min(D) > self.d_max),
        }
