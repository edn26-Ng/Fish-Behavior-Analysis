"""
YOLOv8 detector for the fish-behavior pipeline.
Loads weights from config and returns structured boxes for one frame.
No cross-frame state — tracking is handled by tracker.py, which calls
model.track() on the YOLO object this module owns.
"""

from __future__ import annotations

from ultralytics import YOLO


class YOLODetector:
    """
    Single-frame detection wrapper around a YOLO model.

    Exposes self.model so tracker.py can call model.track() on the same
    object — the Kalman filter state stays in one place.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",          # DEFAULT: stock COCO weights (no fish class). Replace with models/tilapia_v1.pt after fine-tuning.
        conf: float = 0.25,                      # DEFAULT: Ultralytics default. Tune on tank footage (glare/ripple false positives vs missed clumped fish).
        classes: list[int] | None = None,        # DEFAULT: all classes. Use [0] (person) for webcam tests; keep None for the 1-class tilapia model.
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes

    def detect(self, image) -> list[dict]:
        """
        Run plain prediction on one BGR frame (no tracking, no persist).
        Returns list of {bbox: [x1,y1,x2,y2], conf, cls_id, cls_name}.
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "cls_id": cls_id,
                "cls_name": self.model.names[cls_id],
            })
        return detections


def build_detector(cfg: dict) -> YOLODetector:
    """
    Instantiate a YOLODetector from the pipeline config's detector: section.
    New detector types go here only — callers never import YOLODetector directly.
    """
    return YOLODetector(
        model_path=cfg.get("model", "yolov8n.pt"),   # DEFAULT: fallback is stock COCO weights; set detector.model to tilapia weights in config.
        conf=cfg.get("conf", 0.25),                  # DEFAULT: untuned threshold; set detector.conf after evaluating on tank footage.
        classes=cfg.get("classes"),                  # DEFAULT: None = all classes.
    )
