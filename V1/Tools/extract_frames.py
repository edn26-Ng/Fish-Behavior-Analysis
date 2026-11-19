"""
Extract spaced, non-duplicate frames from a recording for detector labeling (YOLO boxes).

Stereo recordings (record_usb.py) are side-by-side (left | right): by default the LEFT eye is
saved, so labels line up with the eye used for detection. Use --eye full for normal videos.

Usage (laptop or Pi):
    python V1/Tools/extract_frames.py --video recordings/..._fish_stereo_seg000.mkv \
        --out datasets/frames/session1 --interval 3
    python V1/Tools/extract_frames.py --video recordings/*.mkv --out datasets/frames/all   # several files

~20 min at --interval 3 → ~400 candidate frames; near-duplicates (fish not moving) are skipped.
Preprocessing goes through Data_Pipeline.preprocess so training frames match inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → `import V1...`
from V1.Data_Pipeline.preprocess import preprocess  # noqa: E402


def pick_eye(img: np.ndarray, eye: str) -> np.ndarray:
    if eye == "full":
        return img
    half = img.shape[1] // 2
    return img[:, :half] if eye == "left" else img[:, half: 2 * half]


def signature(img: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail for near-duplicate checks."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract(video: str, out_dir: Path, interval_s: float, eye: str, min_diff: float, cfg: dict) -> int:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(interval_s * fps))
    stem = Path(video).stem
    saved, idx, last = 0, 0, None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            img, _ = preprocess(pick_eye(frame, eye), cfg)
            sig = signature(img)
            if last is None or float(np.abs(sig - last).mean()) >= min_diff:
                cv2.imwrite(str(out_dir / f"{stem}_f{idx:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last = sig
                saved += 1
        idx += 1
    cap.release()
    print(f"{Path(video).name}: {idx} frames read → {saved} saved")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract labeling frames from recordings")
    ap.add_argument("--video", required=True, nargs="+", help="one or more video files")
    ap.add_argument("--out", required=True, help="output folder for .jpg frames")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between candidates (default 3)")
    ap.add_argument("--eye", choices=["left", "right", "full"], default="left",
                    help="stereo half to save (default left); 'full' for non-stereo video")
    ap.add_argument("--min-diff", type=float, default=2.0,
                    help="skip frames whose thumbnail differs less than this (0-255 scale, default 2)")
    ap.add_argument("--config", help="pipeline config for preprocessing (optional)")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(extract(v, out, args.interval, args.eye, args.min_diff, cfg) for v in args.video)
    print(f"Total: {total} frames in {out.resolve()}")


if __name__ == "__main__":
    main()
