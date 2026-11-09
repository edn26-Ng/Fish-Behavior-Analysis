"""
Replay a tracking CSV back onto the source video.

Draws each track's bounding box and ID onto the corresponding frame.
Use this to verify that the CSV matches what was actually detected.

Usage:
    python V1/Tools/replay.py --video recordings/clip1.mp4 \
        --csv runs/tracks/clip1.csv [--out runs/replay/clip1.mp4] [--no-display]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from V1.Data_Pipeline.source import FileSource


# One stable colour per track_id (cycle through a fixed palette).
_PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112,  31),
    (255, 178, 29),  (207, 210,  49), (72,  249, 10),
    (146, 204,  23), (61,  219, 134), (26,  147,  52),
    (0,  212, 187),  (44,  153, 168), (0,  194, 255),
    (52,   69, 147), (100,  115, 255),(0,   24,  236),
    (132,  56, 255), (82,   0, 133),  (203,  56, 255),
    (255, 149, 200), (255,  55, 199),
]


def _colour(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(track_id) % len(_PALETTE)]


def replay(video_path: str, csv_path: str, out_path: str | None, display: bool) -> None:
    df = pd.read_csv(csv_path)
    # Group detections by frame index for O(1) lookup.
    by_frame: dict[int, pd.DataFrame] = {idx: grp for idx, grp in df.groupby("frame")}

    writer = None

    with FileSource(video_path) as src:
        for frame in src:
            canvas = frame.image.copy()
            rows = by_frame.get(frame.index)

            if rows is not None:
                for _, row in rows.iterrows():
                    x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                    tid = int(row.track_id)
                    colour = _colour(tid)

                    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                    label = f"id:{tid} {row.cls} {row.conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
                    cv2.putText(
                        canvas, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                    )

            cv2.putText(
                canvas, f"frame {frame.index}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
            )

            if out_path is not None:
                if writer is None:
                    h, w = canvas.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(out_path, fourcc, src.fps, (w, h))
                writer.write(canvas)

            if display:
                cv2.imshow("Replay", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    if writer is not None:
        writer.release()
        print(f"Saved: {out_path}")
    if display:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay tracking CSV onto video")
    parser.add_argument("--video", required=True, help="Source video file")
    parser.add_argument("--csv",   required=True, help="Tracking CSV from track_video.py")
    parser.add_argument("--out",   default=None,  help="Save overlay video to this path")
    parser.add_argument("--no-display", action="store_true", help="Skip live window")
    args = parser.parse_args()

    replay(
        video_path=args.video,
        csv_path=args.csv,
        out_path=args.out,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
