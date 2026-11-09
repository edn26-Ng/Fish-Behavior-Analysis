"""
Tracker for the fish-behavior pipeline.
Calls model.track() on every frame sequentially — no drops, no threading.
Uses the YOLO model owned by YOLODetector so Kalman state stays in one object.
"""

from __future__ import annotations

from .detector import YOLODetector


class Tracker:
    """
    Wraps Ultralytics BoT-SORT / ByteTrack tracking on a shared YOLO model.

    Must be called on every frame in order — skipping frames breaks the
    Kalman filter state and causes ID switches and spurious speed spikes.
    """

    # DEFAULT: tracker_cfg = stock Ultralytics BoT-SORT (tuned for pedestrians, camera-motion compensation on).
    # Replace with configs/tracker_fish.yaml tuned on tank footage (gmc_method: none, track_buffer scaled to FPS).
    def __init__(self, detector: YOLODetector, tracker_cfg: str = "botsort.yaml") -> None:
        self._detector = detector
        self._tracker_cfg = tracker_cfg

    def track(self, image) -> list[dict]:
        """
        Run tracking on one BGR frame.
        Returns list of {bbox: [x1,y1,x2,y2], conf, cls_id, cls, track_id}.
        track_id is None when the tracker cannot assign an ID (e.g. first frame,
        low-confidence box) — callers should filter these before logging.
        """
        results = self._detector.model.track(
            image,
            conf=self._detector.conf,
            classes=self._detector.classes,
            persist=True,
            tracker=self._tracker_cfg,
            verbose=False,
        )[0]

        ids = results.boxes.id
        tracks = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            tracks.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "cls_id": cls_id,
                "cls": self._detector.model.names[cls_id],
                "track_id": int(ids[i]) if ids is not None else None,
            })
        return tracks


def build_tracker(detector: YOLODetector, cfg: dict) -> Tracker:
    """
    Instantiate a Tracker from the pipeline config.
    cfg["tracker"] is the path to the tracker YAML (botsort.yaml / tracker_fish.yaml).
    """
    return Tracker(
        detector=detector,
        tracker_cfg=cfg.get("tracker", "botsort.yaml"),  # DEFAULT: stock BoT-SORT fallback; point config `tracker:` at configs/tracker_fish.yaml.
    )
