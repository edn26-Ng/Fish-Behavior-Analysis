"""
CSV logger for the fish-behavior pipeline.
Writes one row per tracked object per frame.
Coordinates are always in original-frame pixels (crop offset applied by caller).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any


_COLUMNS = [
    "video",
    "frame",
    "timestamp_s",
    "track_id",
    "x1", "y1", "x2", "y2",
    "cx", "cy",
    "conf",
    "cls",
]


class CSVLogger:
    """
    Open once per run, call write_frame() for each frame, close() when done.
    Supports context-manager usage.

    Args:
        csv_path:    Destination .csv file. Parent directories are created.
        video_name:  Stored verbatim in the 'video' column for traceability.
        config_path: If given, a copy is saved next to the CSV for reproducibility.
    """

    def __init__(
        self,
        csv_path: str | Path,
        video_name: str,
        config_path: str | Path | None = None,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._video_name = video_name
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS)
        self._writer.writeheader()

        if config_path is not None:
            dest = self._csv_path.with_suffix(".pipeline.yaml")
            shutil.copy2(config_path, dest)

    def write_frame(
        self,
        frame_index: int,
        timestamp_s: float,
        detections: list[dict[str, Any]],
    ) -> None:
        """
        Write one row per detection.

        Each detection dict must contain:
            bbox      – [x1, y1, x2, y2] in original-frame pixels
            conf      – float confidence
            cls       – class name string
            track_id  – stable integer ID from the tracker
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            self._writer.writerow({
                "video":       self._video_name,
                "frame":       frame_index,
                "timestamp_s": round(timestamp_s, 6),
                "track_id":    det["track_id"],
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "cx": round((x1 + x2) / 2, 2),
                "cy": round((y1 + y2) / 2, 2),
                "conf": round(float(det["conf"]), 4),
                "cls": det["cls"],
            })

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
