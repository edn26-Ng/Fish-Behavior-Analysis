"""
Stereo calibration for the side-by-side USB stereo camera (left | right in one frame).

Record a checkerboard video with the camera held still (V1/Capture/record_usb.py --label calib),
moving the board through all 9 image zones, several distances and tilts. Then on the laptop:

    python V1/Tools/calibrate_stereo.py --video 2026..._calib_stereo_seg000.mkv
    python V1/Tools/calibrate_stereo.py --video calib.mkv --square-mm 25.2   # measured square size

Board: docs/calibration/checkerboard_letter_25mm.pdf → 10x7 squares = 9x6 inner corners.

Writes (default V1/configs/stereo_calib.yaml):
  per-eye camera matrix + distortion, R/T between eyes, rectification (R1 R2 P1 P2 Q),
  baseline_mm, RMS errors, image coverage. Also <out>_check.jpg: rectified pair with
  horizontal lines; the same board corner must sit on the same line in both eyes.

Resumable: board detections are checkpointed to <video>.corners.jsonl as they are found; rerunning
the same command after an interruption (Ctrl-C, SSH drop, power loss) skips frames already done.
Use --fresh to start over. Progress prints every 50 frames.

Pass checks: stereo RMS < ~1.0 px, baseline matches a ruler measurement of the lens gap,
coverage has board detections in all 9 zones of each eye.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK


def board_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return obj


def find_corners(gray: np.ndarray, pattern: tuple[int, int], scale: float = 0.5) -> np.ndarray | None:
    """Search on a downscaled copy (~4x faster), then refine to sub-pixel on the full image."""
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, c = cv2.findChessboardCorners(small, pattern, FIND_FLAGS)
        if not ok:
            return None
        c = (c / scale).astype(np.float32)
    else:
        ok, c = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
        if not ok:
            return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), CRIT)


def sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[max(y, 0): y + h, max(x, 0): x + w]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


def zone_counts(corner_sets: list[np.ndarray], w: int, h: int) -> list[list[int]]:
    """How many detections have their board centre in each cell of a 3x3 grid."""
    grid = [[0] * 3 for _ in range(3)]
    for c in corner_sets:
        cx, cy = c.reshape(-1, 2).mean(axis=0)
        grid[min(int(3 * cy / h), 2)][min(int(3 * cx / w), 2)] += 1
    return grid


def detect_all(video: str, pattern, step: int, scale: float, cache: Path, fresh: bool):
    """Pass 1: find the board in every Nth frame. Results go to a checkpoint file line by line,
    so an interrupted run (Ctrl-C, SSH drop, power loss) resumes where it stopped."""
    done: dict[int, dict] = {}
    if cache.exists() and not fresh:
        for line in cache.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line after a power cut
            done[rec["i"]] = rec
        if done:
            print(f"Resuming: {len(done)} frames already checked (checkpoint {cache.name})", flush=True)
    elif fresh and cache.exists():
        cache.unlink()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    eye_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    eye_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    todo = total // step if total else 0
    with cache.open("a") as out:
        i = 0
        while True:
            i += 1
            if i % step or i in done:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left, right = gray[:, :eye_w], gray[:, eye_w: 2 * eye_w]
            rec = {"i": i, "l": None, "r": None, "sharp": 0.0}
            cl = find_corners(left, pattern, scale)
            cr = find_corners(right, pattern, scale) if cl is not None else None
            if cl is not None and cr is not None:
                rec.update(l=cl.reshape(-1, 2).tolist(), r=cr.reshape(-1, 2).tolist(),
                           sharp=min(sharpness(left, cl), sharpness(right, cr)))
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done[i] = rec
            if len(done) % 50 == 0:
                print(f"  checked {len(done)}/{todo or '?'} frames, board in both eyes: "
                      f"{sum(r['l'] is not None for r in done.values())}", flush=True)
    cap.release()
    return done, (eye_w, eye_h)


def select(done: dict, max_pairs: int, min_move_px: float, min_sharp: float):
    """Pass 2: keep sharp, distinct poses seen in both eyes (same result on every resume)."""
    pairs, last_centre = [], None
    for i in sorted(done):
        rec = done[i]
        if rec["l"] is None or rec["sharp"] < min_sharp:
            continue
        cl = np.array(rec["l"], np.float32).reshape(-1, 1, 2)
        cr = np.array(rec["r"], np.float32).reshape(-1, 1, 2)
        centre = cl.reshape(-1, 2).mean(axis=0)
        if last_centre is not None and np.linalg.norm(centre - last_centre) < min_move_px:
            continue
        last_centre = centre
        pairs.append((cl, cr))
    print(f"Checked {len(done)} frames → {len(pairs)} usable board pairs (seen in both eyes, sharp, distinct)",
          flush=True)
    if len(pairs) > max_pairs:  # keep an even spread through the video
        idx = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[k] for k in idx]
    return pairs


def check_image(video: str, maps, size, out_path: Path) -> None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    w = size[0]
    L = cv2.remap(frame[:, :w], *maps[0], cv2.INTER_LINEAR)
    R = cv2.remap(frame[:, w: 2 * w], *maps[1], cv2.INTER_LINEAR)
    both = np.hstack([L, R])
    for y in range(0, both.shape[0], 40):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(str(out_path), both)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the side-by-side USB stereo camera")
    ap.add_argument("--video", required=True, help="checkerboard recording (.mkv)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="measured square size (default 25)")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--step", type=int, default=3, help="check every Nth frame (default 3)")
    ap.add_argument("--max-pairs", type=int, default=60, help="cap on board views used (default 60)")
    ap.add_argument("--min-move-px", type=float, default=40.0, help="skip poses closer than this (default 40)")
    ap.add_argument("--min-sharp", type=float, default=50.0, help="Laplacian variance blur cutoff (default 50)")
    ap.add_argument("--scale", type=float, default=0.5, help="search on a downscaled image (default 0.5; 1 = full size)")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint and start over")
    ap.add_argument("--out", default="V1/configs/stereo_calib.yaml")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    cache = Path(args.video).with_suffix(".corners.jsonl")  # checkpoint next to the video
    done, size = detect_all(args.video, pattern, args.step, args.scale, cache, args.fresh)
    pairs = select(done, args.max_pairs, args.min_move_px, args.min_sharp)
    if len(pairs) < 10:
        raise SystemExit("Too few usable views (<10). Move slower, pause at each pose, keep the board in both eyes.")

    obj = [board_points(*pattern, args.square_mm)] * len(pairs)
    left = [p[0] for p in pairs]
    right = [p[1] for p in pairs]

    flags = cv2.CALIB_RATIONAL_MODEL  # wide lenses: extra radial terms
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj, left, size, None, None, flags=flags)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(obj, right, size, None, None, flags=flags)
    rms_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj, left, right, K1, D1, K2, D2, size,
        criteria=CRIT, flags=cv2.CALIB_FIX_INTRINSIC | flags,
    )
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    maps = (
        cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1),
        cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1),
    )

    cov_l = zone_counts(left, *size)
    cov_r = zone_counts(right, *size)
    baseline = float(np.linalg.norm(T))
    out = {
        "board": {"inner_corners": list(pattern), "square_mm": args.square_mm},
        "image_size_per_eye": list(size),
        "views_used": len(pairs),
        "rms_px": {"left": round(rms_l, 4), "right": round(rms_r, 4), "stereo": round(rms_s, 4)},
        "baseline_mm": round(baseline, 2),
        "focal_px": round(float(P1[0, 0]), 2),
        "coverage_3x3": {"left": cov_l, "right": cov_r},
        **{k: v.tolist() for k, v in dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, R2=R2,
                                          P1=P1, P2=P2, Q=Q).items()},
        "source_video": Path(args.video).name,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False))
    check_path = out_path.with_name(out_path.stem + "_check.jpg")
    check_image(args.video, maps, size, check_path)

    print(f"\nRMS reprojection error (px): left {rms_l:.3f} | right {rms_r:.3f} | stereo {rms_s:.3f}"
          f"  {'✅' if rms_s < 1.0 else '⚠️ high: re-record slower / flatter board'}")
    print(f"Baseline: {baseline:.1f} mm   (measure the lens gap with a ruler to confirm)")
    print(f"Rectified focal length: {P1[0, 0]:.1f} px")
    for name, g in (("left", cov_l), ("right", cov_r)):
        empty = sum(v == 0 for row in g for v in row)
        print(f"Coverage {name} (3x3): {g}  {'✅' if empty == 0 else f'⚠️ {empty} empty zone(s)'}")
    print(f"\nSaved {out_path} and {check_path}")


if __name__ == "__main__":
    main()
