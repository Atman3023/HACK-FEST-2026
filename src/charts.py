"""Static PNG charts for reports/accuracy.md (light surface, reference palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SHORT = {"BLOOD_PRESSURE": "BP", "STOMACH_ACHE": "STOMACH"}


def _short(c):
    return SHORT.get(c, c)


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def confusion(cm, labels, path, title):
    cm = np.asarray(cm, float)
    rows = cm.sum(1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = _fig(1.3 * len(labels) + 2.2, 1.1 * len(labels) + 1.6)
    _style(ax, title)
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), [_short(l) for l in labels])
    ax.set_yticks(range(len(labels)), [_short(l) for l in labels])
    ax.set_xlabel("Predicted", color=INK2)
    ax.set_ylabel("True", color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def per_class(acc, counts, path, title):
    labels = list(acc)
    vals = [0 if acc[l] is None else acc[l] for l in labels]
    fig, ax = _fig(7, 0.6 * len(labels) + 1.4)
    _style(ax, title)
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for y, v, l in zip(yy, vals, labels):
        ax.text(v + 0.015, y, "n/a" if acc[l] is None else f"{v:.0%}  (n={counts[l]})",
                va="center", color=INK, fontsize=10)
    ax.set_yticks(yy, [_short(l) for l in labels])
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def bands(per_class_bands, path, title):
    """per_class_bands: {class: {"retry": n, "confirm": n, "final": n}}"""
    labels = list(per_class_bands)
    names = ["retry", "confirm", "final"]
    fig, ax = _fig(7.5, 0.6 * len(labels) + 1.8)
    _style(ax, title)
    yy = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for name, color in zip(names, SERIES):
        tot = np.array([max(sum(per_class_bands[l].values()), 1) for l in labels])
        v = np.array([per_class_bands[l][name] for l in labels]) / tot
        ax.barh(yy, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        for y, lv, vv in zip(yy, left, v):
            if vv >= 0.12:
                ax.text(lv + vv / 2, y, f"{vv:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += v
    ax.set_yticks(yy, [_short(l) for l in labels])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def detection(det, path, title):
    """det: {class: {"body": rate, "hands": rate}}"""
    labels = list(det)
    fig, ax = _fig(7.5, 3.6)
    _style(ax, title)
    x = np.arange(len(labels))
    for k, (key, color) in enumerate(zip(("body", "hands"), SERIES)):
        v = [det[l][key] for l in labels]
        ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=color, edgecolor=SURFACE, linewidth=2, label=key)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.36, vi + 0.02, f"{vi:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [_short(l) for l in labels])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper right", frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
