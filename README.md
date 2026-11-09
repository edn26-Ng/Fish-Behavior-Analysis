# Fish-Behavior-Analysis
Development of fish behavior analysis algorithm via Raspeberry Pi4 stereo vision camera system.

## Model weights
Weights are not committed (see `.gitignore`). They download automatically on first run:
- `yolov8n.pt` / `yolov8s.pt`: Ultralytics fetches them when `YOLO("<name>.pt")` is first loaded.
- `hand_landmarker.task`: `demo/detector.py` downloads it from MediaPipe's model storage.

Fine-tuned tilapia weights will live outside the repo; document the location here once they exist.
