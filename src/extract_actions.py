"""Build the action library: data/actions/<CLASS>/<signer>_<n>.mp4 -> data/library/actions.npz

Run:  python -m src.extract_actions
Every clip's Holistic detection rate is stored (including clips too poor to use).
"""
import sys
from collections import defaultdict

import numpy as np

from src import config
from src.action_features import mirror, process_frames, read_video

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ACTIONS_LIBRARY_PATH = config.ROOT / "data" / "library" / "actions.npz"


def signer_of(path):
    return path.stem.rsplit("_", 1)[0].lower() if "_" in path.stem else path.stem.lower()


def main(actions_dir=None, out_path=None):
    actions_dir = actions_dir or config.ACTIONS_DIR
    out_path = out_path or ACTIONS_LIBRARY_PATH
    clips = [(c, p) for c in config.ACTION_CLASSES if (actions_dir / c).exists()
             for p in sorted((actions_dir / c).iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        print(f"No clips in {actions_dir}. Expected data/actions/<CLASS>/<signer>_<n>.mp4")
        return 1

    S, Sm, y, signers, files = [], [], [], [], []
    stats = defaultdict(list)
    for i, (cls, path) in enumerate(clips, 1):
        frames = read_video(path)
        seq, st = process_frames(frames) if frames else (None, {"frames": 0, "pose_rate": 0.0, "hand_rate": 0.0})
        ok = seq is not None
        for key in ("pose_rate", "hand_rate", "frames"):
            stats[key].append(st[key])
        stats["cls"].append(cls); stats["file"].append(path.name); stats["usable"].append(ok)
        print(f"  [{i}/{len(clips)}] {cls}/{path.name}: {st['frames']} frames, body {st['pose_rate']:.0%}, "
              f"hands {st['hand_rate']:.0%}{'' if ok else '  -> SKIPPED (body not visible enough)'}")
        if ok:
            S.append(seq); Sm.append(mirror(seq)); y.append(cls); signers.append(signer_of(path))
            files.append(f"{cls}/{path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = S[0].shape if S else (0, 0)
    np.savez(out_path,
             S=np.stack(S) if S else np.zeros((0, *dim), np.float32),
             Sm=np.stack(Sm) if Sm else np.zeros((0, *dim), np.float32),
             y=np.array(y), signer=np.array(signers), files=np.array(files),
             stat_cls=np.array(stats["cls"]), stat_file=np.array(stats["file"]),
             stat_pose=np.array(stats["pose_rate"]), stat_hand=np.array(stats["hand_rate"]),
             stat_frames=np.array(stats["frames"]), stat_usable=np.array(stats["usable"]))

    print(f"\nSaved {len(S)}/{len(clips)} usable clips -> {out_path}")
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
