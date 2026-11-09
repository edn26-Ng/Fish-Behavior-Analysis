"""
Extract spaced labeling frames from a video.
Skeleton — interval and near-duplicate threshold need tuning on tilapia recordings.

Usage:
    python V1/Tools/extract_frames.py --video recordings/clip1.mp4 \
        --config configs/pipeline.yaml --out datasets/frames/clip1 [--interval 4.0]
"""

from __future__ import annotations

import argparse


def extract(video_path: str, cfg: dict, out_dir: str, interval_s: float) -> None:
    # TODO: open video with FileSource
    # TODO: for each frame at ~interval_s spacing:
    #         apply preprocess() (same as pipeline — training/inference must match)
    #         compute a near-duplicate hash (e.g. perceptual hash or frame diff)
    #         skip if too similar to the last saved frame
    #         write image to out_dir with zero-padded filename (frame_{:06d}.jpg)
    # TODO: print count of saved frames
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract labeling frames from video")
    parser.add_argument("--video",    required=True, help="Source video file")
    parser.add_argument("--config",   required=True, help="Pipeline config (for preprocessing)")
    parser.add_argument("--out",      required=True, help="Output directory for frames")
    parser.add_argument("--interval", type=float, default=4.0,
                        help="Minimum seconds between saved frames (default: 4.0)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    extract(args.video, cfg, args.out, args.interval)


if __name__ == "__main__":
    main()
