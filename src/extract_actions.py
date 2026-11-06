"""Build the action library: data/actions/<CLASS>/<signer>_<n>.mp4 -> data/library/actions.npz

Run:  python -m src.extract_actions
Every clip's Holistic detection rate is stored (including clips too poor to use).
Augmented copies (scripts/segment_videos.py) are named <signer>_<n>~aug-<kind>.mp4; they are stored with
aug=True so evaluation never uses them as test clips.
Per clip the library also stores its parent segment (original clip it was copied from) and its source
video (from data/actions/segments_manifest.csv; a directly recorded clip is its own source).
Features are cached per file (data/library/action_cache.npz, keyed on size + modification time), so a
rebuild only runs MediaPipe on new or changed clips.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import config
from src.action_features import mirror, process_frames, read_video

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ACTIONS_LIBRARY_PATH = config.ROOT / "data" / "library" / "actions.npz"
CACHE_PATH = config.ROOT / "data" / "library" / "action_cache.npz"
MANIFEST_PATH = config.ACTIONS_DIR / "segments_manifest.csv"
AUG_MARK = "~aug-"   # <signer>_<n>~aug-<kind>.mp4 = augmented copy of <signer>_<n>.mp4 (training only)


def is_augmented(path):
    return AUG_MARK in path.stem


def signer_of(path):
    stem = path.stem.split(AUG_MARK)[0]
    return stem.rsplit("_", 1)[0].lower() if "_" in stem else stem.lower()


def parent_of(cls, path):
    """Original clip an (augmented) clip comes from, e.g. 'FEVER/p1_2'."""
    return f"{cls}/{path.stem.split(AUG_MARK)[0]}"


def load_sources(manifest=MANIFEST_PATH):
    """{'CLASS/file.mp4': 'source video'} from the segmentation manifest."""
    if not Path(manifest).exists():
        return {}
    with open(manifest, newline="") as fh:
        return {r["file"]: r["source"] for r in csv.DictReader(fh)}


def load_cache(path=CACHE_PATH):
    if not path.exists():
        return {}
    c = np.load(path)
    out = {}
    for i, f in enumerate(c["files"]):
        seq = c["seq"][i]
        out[str(f)] = {"mtime": float(c["mtime"][i]), "size": int(c["size"][i]),
                       "seq": None if np.isnan(seq).all() else seq,
                       "st": {"frames": int(c["frames"][i]), "pose_rate": float(c["pose"][i]),
                              "hand_rate": float(c["hand"][i])}}
    return out


def save_cache(cache, path=CACHE_PATH):
    from src.action_features import DIM, N_FRAMES
    keys = sorted(cache)
    nan = np.full((N_FRAMES, DIM), np.nan, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, files=np.array(keys), mtime=np.array([cache[k]["mtime"] for k in keys]),
             size=np.array([cache[k]["size"] for k in keys]),
             seq=np.stack([nan if cache[k]["seq"] is None else cache[k]["seq"] for k in keys]) if keys
             else np.zeros((0, N_FRAMES, DIM), np.float32),
             frames=np.array([cache[k]["st"]["frames"] for k in keys]),
             pose=np.array([cache[k]["st"]["pose_rate"] for k in keys]),
             hand=np.array([cache[k]["st"]["hand_rate"] for k in keys]))


def main(actions_dir=None, out_path=None):
    actions_dir = actions_dir or config.ACTIONS_DIR
    out_path = out_path or ACTIONS_LIBRARY_PATH
    clips = [(c, p) for c in config.ACTION_CLASSES if (actions_dir / c).exists()
             for p in sorted((actions_dir / c).iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        print(f"No clips in {actions_dir}. Expected data/actions/<CLASS>/<signer>_<n>.mp4")
        return 1

    S, Sm, y, signers, files, aug, parents, sources = [], [], [], [], [], [], [], []
    stats = defaultdict(list)
    cache, manifest, n_new = load_cache(), load_sources(), 0
    for i, (cls, path) in enumerate(clips, 1):
        key, fs = f"{cls}/{path.name}", path.stat()
        hit = cache.get(key)
        if hit and hit["size"] == fs.st_size and abs(hit["mtime"] - fs.st_mtime) < 1e-3:
            seq, st = hit["seq"], hit["st"]
        else:
            frames = read_video(path)
            seq, st = process_frames(frames) if frames else (None, {"frames": 0, "pose_rate": 0.0, "hand_rate": 0.0})
            cache[key] = {"mtime": fs.st_mtime, "size": fs.st_size, "seq": seq, "st": st}
            n_new += 1
        ok = seq is not None
        for stat in ("pose_rate", "hand_rate", "frames"):
            stats[stat].append(st[stat])
        stats["cls"].append(cls); stats["file"].append(path.name); stats["usable"].append(ok)
        print(f"  [{i}/{len(clips)}] {cls}/{path.name}: {st['frames']} frames, body {st['pose_rate']:.0%}, "
              f"hands {st['hand_rate']:.0%}{'' if ok else '  -> SKIPPED (body not visible enough)'}")
        if ok:
            S.append(seq); Sm.append(mirror(seq)); y.append(cls); signers.append(signer_of(path))
            files.append(key); aug.append(is_augmented(path)); parents.append(parent_of(cls, path))
            sources.append(manifest.get(key) or manifest.get(f"{parent_of(cls, path)}.mp4") or f"{parent_of(cls, path)}.mp4")
    present = {f"{c}/{p.name}" for c, p in clips}
    save_cache({k: v for k, v in cache.items() if k in present})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = S[0].shape if S else (0, 0)
    np.savez(out_path,
             S=np.stack(S) if S else np.zeros((0, *dim), np.float32),
             Sm=np.stack(Sm) if Sm else np.zeros((0, *dim), np.float32),
             y=np.array(y), signer=np.array(signers), files=np.array(files), aug=np.array(aug, bool),
             parent=np.array(parents), source=np.array(sources),
             stat_cls=np.array(stats["cls"]), stat_file=np.array(stats["file"]),
             stat_pose=np.array(stats["pose_rate"]), stat_hand=np.array(stats["hand_rate"]),
             stat_frames=np.array(stats["frames"]), stat_usable=np.array(stats["usable"]))

    print(f"\nFeatures: {n_new} clip(s) extracted, {len(clips) - n_new} from cache")
    print(f"Saved {len(S)}/{len(clips)} usable clips ({sum(aug)} augmented copies, training only) -> {out_path}")
    print(f"{'class':<16}{'clips':>6}{'usable':>8}{'body':>7}{'hands':>7}  signers")
    sc, su = np.array(stats["cls"]), np.array(stats["usable"])
    for c in config.ACTION_CLASSES:
        m = sc == c
        if m.any():
            sig = sorted({s for s, cc in zip(signers, y) if cc == c})
            print(f"{c:<16}{m.sum():>6}{su[m].sum():>8}{np.mean(np.array(stats['pose_rate'])[m]):>7.0%}"
                  f"{np.mean(np.array(stats['hand_rate'])[m]):>7.0%}  {', '.join(sig)}")
    return 0 if S else 1


if __name__ == "__main__":
    sys.exit(main())
