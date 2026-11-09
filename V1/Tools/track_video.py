"""
trackerCLI — run the full pipeline on one video file.

Usage:
    python V1/Tools/track_video.py --config configs/pipeline.yaml \
        --video recordings/clip1.mp4 --out runs/tracks/clip1.csv

CLI args override the values in the config file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from V1.Data_Pipeline.source import build_source
from V1.Data_Pipeline.preprocess import preprocess
from V1.Data_Pipeline.detector import build_detector
from V1.Data_Pipeline.tracker import build_tracker
from V1.Data_Pipeline.logger import CSVLogger


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.video:
        cfg.setdefault("source", {})["path"] = args.video
        cfg["source"]["type"] = "file"
    if args.out:
        cfg.setdefault("output", {})["path"] = args.out


def run(cfg: dict, config_path: str) -> None:
    out_path = Path(
        cfg.get("output", {}).get("path")
        or Path(cfg["output"]["dir"]) / (Path(cfg["source"]["path"]).stem + ".csv")
    )
    video_name = Path(cfg["source"]["path"]).name

    detector = build_detector(cfg["detector"])
    tracker = build_tracker(detector, cfg)

    t0 = time.perf_counter()
    frame_count = 0
    det_count = 0

    with build_source(cfg["source"]) as src, \
         CSVLogger(out_path, video_name, config_path=config_path) as log:

        for frame in src:
            image, crop_offset = preprocess(frame.image, cfg.get("preprocess", {}))
            detections = tracker.track(image)

            # Shift bboxes back to original-frame pixels using the crop offset.
            ox, oy = crop_offset
            if ox or oy:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    det["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]

            tracked = [d for d in detections if d["track_id"] is not None]
            log.write_frame(frame.index, frame.timestamp, tracked)

            frame_count += 1
            det_count += len(tracked)

            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - t0
                fps = frame_count / elapsed
                print(f"  frame {frame_count:>6}  |  {fps:.1f} fps  |  {det_count} detections so far")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"Output: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish behavior tracker CLI")
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    parser.add_argument("--video", default=None, help="Override source video path")
    parser.add_argument("--out", default=None, help="Override output CSV path")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _apply_overrides(cfg, args)
    run(cfg, args.config)


if __name__ == "__main__":
    main()
