"""
Tracking-quality report.
Skeleton — thresholds and jump detection need real fish data to calibrate.

Usage:
    python V1/Tools/track_report.py --csv runs/tracks/clip1.csv [--fish-count 75]
"""

from __future__ import annotations

import argparse


def load_csv(csv_path: str):
    # TODO: load tracking CSV into a DataFrame
    raise NotImplementedError


def report_id_count(df, expected_fish: int | None) -> None:
    # TODO: count unique track_ids; compare to expected_fish if provided
    raise NotImplementedError


def report_track_lengths(df) -> None:
    # TODO: distribution of per-ID frame counts; flag IDs shorter than ~1 s
    raise NotImplementedError


def report_fragmentation(df) -> None:
    # TODO: detect IDs that appear, disappear, and reappear (fragmented tracks)
    raise NotImplementedError


def report_impossible_jumps(df) -> None:
    # TODO: flag per-ID frame-to-frame centroid jumps above a threshold (px/frame)
    #       threshold needs calibration from fish data
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracking quality report")
    parser.add_argument("--csv", required=True, help="Tracking CSV from track_video.py")
    parser.add_argument("--fish-count", type=int, default=None,
                        help="Expected number of fish (for ID-count comparison)")
    args = parser.parse_args()

    df = load_csv(args.csv)
    report_id_count(df, args.fish_count)
    report_track_lengths(df)
    report_fragmentation(df)
    report_impossible_jumps(df)


if __name__ == "__main__":
    main()
