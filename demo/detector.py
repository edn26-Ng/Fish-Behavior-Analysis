# Detection module — wraps YOLOv8n inference behind a stable interface.
# Swap the model path in config to move from pretrained COCO to a
# fine-tuned fish model without touching any other file.

import os
import queue
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# MediaPipe 1.x Tasks API model — downloaded once and cached locally.
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_MODEL_PATH = "hand_landmarker.task"


# Loads a YOLO model and runs per-frame inference.
# Returns structured detections so callers never touch raw YOLO result objects.
class YOLODetector:

    # tracker: "botsort.yaml" enables BoT-SORT; None falls back to plain predict().
    # persist=True keeps Kalman filter state alive between calls on the same thread.
    def __init__(self, model_path: str = "yolov8s.pt", conf: float = 0.25,
                 classes: list[int] | None = None, tracker: str | None = "botsort.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes
        self.tracker = tracker

    # Runs tracking (or plain detection) on a single BGR frame.
    # Each dict: {bbox, conf, cls_id, cls_name, track_id} — track_id is None without tracker.
    def detect(self, frame) -> list[dict]:
        if self.tracker:
            results = self.model.track(
                frame,
                conf=self.conf,
                classes=self.classes,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )[0]
        else:
            results = self.model.predict(
                frame, conf=self.conf, classes=self.classes, verbose=False,
            )[0]

        ids = results.boxes.id
        detections = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "cls_id": cls_id,
                "cls_name": self.model.names[cls_id],
                "track_id": int(ids[i]) if ids is not None else None,
            })
        return detections


# Uses MediaPipe HandLandmarker (Tasks API, mediapipe>=1.x) to detect hands.
# Downloads hand_landmarker.task on first run; cached locally afterward.
# Same detect() interface as YOLODetector — fully swappable.
class HandDetector:

    # max_hands: maximum simultaneous hands. conf: detection confidence gate.
    # VIDEO mode enables MediaPipe's internal Kalman tracking between frames,
    # which significantly reduces per-frame latency vs IMAGE mode.
    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5):
        if not os.path.exists(_HAND_MODEL_PATH):
            print("Downloading MediaPipe hand model...")
            urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.monotonic() * 1000)

    # Returns one detection dict per hand found, with bbox padded 10 px each side.
    def detect(self, frame) -> list[dict]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000) - self._start_ms
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections = []
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]
            pad = 10
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": 1.0,
                "cls_id": 0,
                "cls_name": "hand",
                "track_id": idx,  # MediaPipe VIDEO mode preserves hand order stably
            })
        return detections


# Runs YOLODetector on a background thread so the main capture loop is never
# blocked by inference. The main loop submits frames non-blocking and reads
# the latest available results independently of inference speed.
class ThreadedDetector:

    def __init__(self, detector: YOLODetector):
        self._detector = detector
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._results: list[dict] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            dets = self._detector.detect(frame)
            with self._lock:
                self._results = dets

    # Puts the frame in the queue for inference; drops it if the worker is busy.
    def submit(self, frame):
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    # Returns the most recent completed detection results.
    def latest_detections(self) -> list[dict]:
        with self._lock:
            return list(self._results)

    # Signals the worker thread to exit cleanly.
    def stop(self):
        self._queue.put(None)
        self._thread.join()


# 12 visually distinct BGR colors — chosen so no two adjacent entries look similar.
_PALETTE = [
    ( 60, 180, 255),  # orange
    ( 60, 220,  60),  # green
    (255,  60,  60),  # blue
    (200,  60, 220),  # purple
    ( 40, 220, 220),  # yellow
    (255,  60, 160),  # pink
    (200, 200,  40),  # cyan
    ( 50,  50, 220),  # red
    (140, 255, 100),  # lime
    (255, 100,  40),  # sky blue
    ( 60, 160, 255),  # amber
    (180,  60, 255),  # magenta
]

# Maps (class ID, within-class instance index) to a palette color.
# Prime multipliers ensure no two class IDs share an entry even with many classes.
def instance_color(cls_id: int, instance_idx: int) -> tuple[int, int, int]:
    return _PALETTE[(cls_id * 7 + instance_idx * 3) % len(_PALETTE)]


# Draws bounding boxes and labels onto the frame in-place.
# When a track_id is present (Stage 3+), color and label are keyed to the track
# so each object keeps a consistent color across frames. Without a track_id,
# falls back to per-class instance coloring.
def draw_detections(frame, detections: list[dict]):
    class_counts: dict[int, int] = {}
    for det in detections:
        track_id = det.get("track_id")
        cls_id = det["cls_id"]
        x1, y1, x2, y2 = det["bbox"]

        if track_id is not None:
            color = _PALETTE[track_id % len(_PALETTE)]
            label = f"#{track_id} {det['cls_name']} {det['conf']:.2f}"
        else:
            idx = class_counts.get(cls_id, 0)
            class_counts[cls_id] = idx + 1
            color = instance_color(cls_id, idx)
            label = f"{det['cls_name']} {det['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# Returns a ThreadedDetector wrapping either HandDetector or YOLODetector
# depending on the mode field in config.yaml. Switching modes requires only
# a config change — no code changes elsewhere.
def build_detector(cfg: dict) -> ThreadedDetector:
    mode = cfg.get("mode", "yolo")
    if mode == "hand":
        base = HandDetector(
            max_hands=cfg.get("max_hands", 2),
            min_detection_confidence=cfg.get("conf", 0.5),
        )
    else:
        base = YOLODetector(
            model_path=cfg.get("model", "yolov8s.pt"),
            conf=cfg.get("conf", 0.25),
            classes=cfg.get("classes", None),
            tracker=cfg.get("tracker", "botsort.yaml"),
        )
    return ThreadedDetector(base)
