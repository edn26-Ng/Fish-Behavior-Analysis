"""
Frame sources for the fish-behavior pipeline.
All sources produce Frame objects with identical fields so downstream
stages never need to know which source is in use.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray   # BGR uint8
    index: int          # 0-based frame counter
    timestamp: float    # seconds


class BaseSource:
    """Abstract base — subclasses implement read() and release()."""

    def read(self) -> Frame | None:
        """Return the next Frame, or None when the stream is exhausted."""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> "BaseSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()


class FileSource(BaseSource):
    """
    Read from a video file.
    timestamp = index / fps — processing speed never distorts measured fish speed.
    fps_override is required for raw Pi .h264 files (no FPS metadata).
    """

    def __init__(self, path: str, fps_override: float | None = None) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video file: {path}")

        if fps_override is not None:
            self._fps = float(fps_override)
        else:
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            if not self._fps or self._fps <= 0:
                raise ValueError(
                    f"File '{path}' has no FPS metadata. "
                    "Pass fps_override= (e.g. fps_override=30) or convert to .mp4 first."
                )

        self._index = 0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        frame = Frame(
            image=image,
            index=self._index,
            timestamp=self._index / self._fps,
        )
        self._index += 1
        return frame

    def release(self) -> None:
        self._cap.release()


class WebcamSource(BaseSource):
    """
    Live webcam — quick tests only.
    Uses CAP_DSHOW on Windows (avoids long probe delays), CAP_ANY elsewhere.
    Timestamps are wall-clock elapsed seconds from the first frame.
    """

    def __init__(self, device_index: int = 0) -> None:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(device_index, backend)
        if not self._cap.isOpened():
            raise OSError(f"Cannot open webcam at index {device_index}")
        self._index = 0
        self._t0: float | None = None

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        frame = Frame(
            image=image,
            index=self._index,
            timestamp=now - self._t0,
        )
        self._index += 1
        return frame

    def release(self) -> None:
        self._cap.release()


class PicameraSource(BaseSource):
    """Raspberry Pi camera via picamera2. Not implemented until Pi model is confirmed."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "PicameraSource is blocked on the Pi 4 + stereo HAT vs Pi 5 decision. "
            "Use FileSource on recorded .mp4 files for now."
        )

    def read(self) -> Frame | None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


def build_source(cfg: dict) -> BaseSource:
    """
    Instantiate the correct source from the pipeline config dict.
    cfg mirrors configs/pipeline.yaml  source: section.

    New source types must be added here only — callers never import
    FileSource / WebcamSource / PicameraSource directly.
    """
    src_type = cfg.get("type", "file").lower()

    if src_type == "file":
        return FileSource(
            path=cfg["path"],
            fps_override=cfg.get("fps_override"),
        )

    if src_type == "webcam":
        return WebcamSource(device_index=cfg.get("device_index", 0))

    if src_type == "picamera2":
        return PicameraSource()

    raise ValueError(f"Unknown source type '{src_type}'. Valid: file, webcam, picamera2.")
