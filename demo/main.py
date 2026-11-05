"""
Stage 2 — YOLOv8n detection on webcam feed.
Pass check: stable bounding boxes drawn, ≥15 FPS on laptop.

Usage:
    python demo/main.py
    python demo/main.py --config demo/config.yaml --source-index 1
"""

import argparse
import time

import cv2
import yaml

from detector import build_detector, draw_detections
from source import build_source


# Defines --config (YAML path) and --source-index (webcam override).
# CLI flags let you switch cameras or configs without editing any file.
def parse_args():
    p = argparse.ArgumentParser(description="Fish Behavior Demo — Stage 1")
    p.add_argument("--config", default="demo/config.yaml")
    p.add_argument("--source-index", type=int, default=None,
                   help="Override webcam index from config")
    return p.parse_args()


# Reads config.yaml and returns it as a plain dict.
# All tunable values (source type, resolution, display options) live here
# so nothing is hard-coded in the logic below.
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Burns the instant and rolling-average FPS into the top-left corner of
# the frame before it is displayed. Instant FPS shows per-frame spikes;
# avg smooths the noise so the number is readable at a glance.
def draw_fps(frame, fps: float, avg_fps: float):
    text = f"FPS: {fps:5.1f}  avg: {avg_fps:5.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


# Draws a two-button class filter panel in the bottom-right corner.
# Semi-transparent dark background makes it readable over any video feed.
# Active button is highlighted green; inactive is dark grey.
def draw_filter_ui(frame, filter_active: bool, configured_filter: list[str] | None):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 1
    pad = 10
    gap = 8
    margin = 14

    filter_label = " | ".join(configured_filter) if configured_filter else "—"
    btn_labels = ["ALL", filter_label]
    active_idx = 1 if filter_active else 0

    sizes = [cv2.getTextSize(l, font, scale, thick)[0] for l in btn_labels]
    btn_h = max(s[1] for s in sizes) + 2 * pad
    btn_ws = [s[0] + 2 * pad for s in sizes]

    hint = "[F] class filter"
    hint_size = cv2.getTextSize(hint, font, 0.5, 1)[0]

    panel_w = max(sum(btn_ws) + gap, hint_size[0]) + 2 * pad
    panel_h = hint_size[1] + gap + btn_h + 2 * pad
    px = w - panel_w - margin
    py = h - panel_h - margin

    # Semi-transparent dark panel background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    # Hint label.
    cv2.putText(frame, hint, (px + pad, py + pad + hint_size[1]),
                font, 0.5, (180, 180, 180), thick, cv2.LINE_AA)

    # Buttons.
    bx = px + pad
    by = py + pad + hint_size[1] + gap
    for i, (label, (tw, th)) in enumerate(zip(btn_labels, sizes)):
        bx1, by1 = bx, by
        bx2, by2 = bx + btn_ws[i], by + btn_h
        is_active = (i == active_idx)
        fill   = (40, 160, 40)   if is_active else (50, 50, 50)
        border = (255, 255, 255) if is_active else (100, 100, 100)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), fill, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border, 1)
        cv2.putText(frame, label, (bx1 + pad, by2 - pad),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        bx = bx2 + gap


# Orchestrates setup and runs the capture loop.
# Uses build_source() as a context manager so the webcam is always
# released cleanly on exit, even if the loop raises an exception.
def main():
    args = parse_args()
    cfg = load_config(args.config)

    src_cfg = cfg["source"]
    if args.source_index is not None:
        src_cfg["webcam_index"] = args.source_index

    win_name = cfg["display"]["window_name"]
    display_w = cfg["display"].get("display_width", 1280)
    display_h = cfg["display"].get("display_height", 720)

    # Two detectors run simultaneously on background threads.
    # ALL mode reads from yolo_detector; hand filter reads from hand_detector.
    yolo_detector = build_detector(cfg["detector"])
    hand_detector = build_detector(cfg.get("hand_detector", {"mode": "hand"}))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_w, display_h)

    configured_filter: list[str] | None = cfg["display"].get("filter_classes")
    filter_active = False  # start in ALL mode

    with build_source(src_cfg) as src:
        print(f"Webcam opened. Reported FPS: {src.actual_fps:.1f}")
        print("Press 'q' to quit.")

        fps_history: list[float] = []
        prev_time = time.perf_counter()

        while True:
            ok, frame = src.read()
            if not ok:
                print("Frame grab failed — exiting.")
                break

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0

            fps_history.append(instant_fps)
            if len(fps_history) > 60:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            if cfg["display"].get("flip_horizontal"):
                frame = cv2.flip(frame, 1)

            # Submit to both detectors every frame; display from the active one.
            yolo_detector.submit(frame)
            hand_detector.submit(frame)
            detections = hand_detector.latest_detections() if filter_active \
                else yolo_detector.latest_detections()

            # Resize to display resolution; draw all overlays on the display
            # frame so coordinates match the visible window size exactly.
            display = cv2.resize(frame, (display_w, display_h))
            sx = display_w / frame.shape[1]
            sy = display_h / frame.shape[0]
            scaled = [
                {**d, "bbox": [int(d["bbox"][0]*sx), int(d["bbox"][1]*sy),
                                int(d["bbox"][2]*sx), int(d["bbox"][3]*sy)]}
                for d in detections
            ]
            draw_detections(display, scaled)

            if cfg["display"]["show_fps"]:
                draw_fps(display, instant_fps, avg_fps)

            draw_filter_ui(display, filter_active, configured_filter)

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
            elif key == ord("f"):
                filter_active = not filter_active

    yolo_detector.stop()
    hand_detector.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
