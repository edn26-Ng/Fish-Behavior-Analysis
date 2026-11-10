# CLAUDE.md — Fish Behavior Analysis

## How to respond
- Answer immediately, no preamble or restating the question.
- State conclusions directly; don't narrate alternatives.
- No filler or hedging. Minimum tokens that stay correct.
- Code or direct answer first; explain only non-obvious details.
- Optimize for ADHD: short sections, bullets, tables, bold key points.

## Keep this file current (standing rule)
- **Every PR that adds or changes a module, config, script or hardware decision must update this file** in the same PR: what it does, how to run it, and any new gotchas.
- Fuller design notes live in `docs/code-environment-organization.md`; update its repo-layout and deferred tables too.

## Project in one paragraph
Hand-labeled video library of tilapia group behavior → train a model that classifies group state → later drive an automated feeder that minimizes feed waste. Pi = capture only for now; the feeder is a purchased, secondary project.

More detail: `docs/project-description.md` (living spec) and `docs/research-notes.md` (literature + budget).

## Labels
1. Resting 2. Hungry 3. Feeding 4. Satiated
Event tag: **Disturbance** (human/lab activity, catches false positives).

## Signals the code must compute
- **Vertical position** of the school (hungry rises, satiated sinks). Needs depth → stereo.
- **Clumping/dispersion**: dispersion around centroid + Delaunay mean edge length (aggregation index).
- **Per-fish kinematics** (a few tracked fish): displacement, speed, acceleration, vertical + horizontal.

## Tank + hardware
- Concrete cylinder, opaque, open top → cameras top-down. Ø ~1.2–1.3 m, depth ~1.2 m, clear water.
- Tilapia, est. 50–100.
- 2× Camera Module 3 Wide (IMX708), top-down stereo, ~69 cm (27") above water (water surface is ~29" below the tank rim), ~10 cm baseline. Short image axis covers ~91 cm of the ~125 cm tank → ~84% of the surface is in frame; two thin edge strips are cut off.
- ⚠️ **Unresolved:** project spec says Pi 4 + stereo HAT; order sheet lists Pi 5 (two native camera ports). **Pi 4 is on hand** → single-camera recording works now via `V1/Capture/record.py`; stereo needs the Pi 5 (or HAT).
- Night recording open: candidate 940 nm IR + NoIR cameras.

## Pi capture (`V1/Capture/record.py`) 🟡
Records on the Pi; all processing happens offline on a laptop.
- **Install (Pi):** `sudo apt install -y python3-picamera2 ffmpeg`. picamera2 is apt-only; a venv needs `--system-site-packages`.
- **Check camera:** `rpicam-hello --list-cameras` → should list `imx708_wide`.
- **Record:** `python3 V1/Capture/record.py --config V1/configs/capture.yaml --duration 60 --label test`
  - `--duration 0` = until Ctrl-C. `--cameras 0 1` = stereo (**Pi 5 only**; Pi 4 has one port).
- **Output per segment, per camera** (in `recordings/`, gitignored):
  - `<stamp>_<label>_cam<N>_seg<K>.mp4`: H.264, 1536×864 @ 30 fps (binned mode = full wide FOV)
  - `…_frames.csv`: one row per **encoded** frame (`frame, sensor_ts_us, wall_time`). Rows == video frames. Use `sensor_ts_us` to pair stereo frames (no hardware sync).
  - `…json`: settings + the locked exposure / gain / white balance
- **Behavior:** auto-exposure settles for `warmup_s`, then exposure, gain and white balance are **locked** for the session; focus is fixed (`lens_position` in dioptres). New file every `segment_s`; stops when free disk < `min_free_gb`.
- **Download:** `scp <user>@<pi>.local:~/Fish-Behavior-Analysis/recordings/* .` Cornell Wi-Fi often blocks `.local`/device SSH → use a phone hotspot or a USB stick.
- **Defaults (`# DEFAULT:` in `capture.yaml`):** `cameras: [0]`, `fps: 30`, `exposure_us/analogue_gain: null`, `lens_position: 1.0`.
- **Gotchas:** mp4 container timestamps come from ffmpeg's wall clock; for kinematics trust `_frames.csv`. Hardware-tested only against a mock so far; verify on the Pi.

## Current task: webcam demo (validate pipeline before Pi/tank)
Modules must stay separate and swappable:

| Module | Demo | Final |
|---|---|---|
| Source | Laptop webcam (OpenCV) | Pi stereo cams / recorded files |
| Detector | YOLOv8n pretrained COCO | YOLOv8 fine-tuned on tilapia |
| Tracker | BoT-SORT (Ultralytics built-in) | Same, or FishSort if code released |
| Features | Per-ID speed/accel + group dispersion | + stereo depth |
| Logger | CSV + raw video | + labels |

### Stages (each has a pass check)
1. **Env**: Python 3.10+, venv, `ultralytics opencv-python numpy pandas scipy`. ✅ webcam window + FPS shown.
2. **Detection**: YOLOv8n on `person` / handheld object. ✅ stable boxes, ≥15 FPS on laptop.
3. **Tracking**: `persist=True`, BoT-SORT. Stress-test occlusion + crossings. ✅ count ID switches.
4. **Features**: centroid history per ID → finite differences. **Smooth before differentiating** (Savitzky–Golay). Group dispersion + Delaunay edge length. Overlay on screen. ✅ values rise with motion.
5. **Logging**: per-frame CSV (timestamp, frame, id, bbox, cx, cy, speed, accel, group metrics) + raw video at full FPS. ✅ replay matches CSV.
6. **Fish video**: COCO has no fish → label ~300–500 frames (DarkLabel/CVAT/Roboflow), fine-tune YOLOv8n. ✅ works on held-out clip.
7. **Pi port**: Picamera2 source. Pi records, laptop processes offline (YOLO on Pi CPU is only a few FPS).

### Gotchas
- Speed is px/frame → convert with FPS (time) and in-water calibration (cm). Top-down pixel scale changes with depth.
- ID switches create fake acceleration spikes → drop/flag tracks shorter than N frames.
- Keep full-frame-rate video even though labels are sparse (~every 3 s); kinematics need consecutive frames.

## Conventions
- Work on feature branches; PRs into `main`.
- Config (source, model, tracker, thresholds) in a YAML/CLI args, not hard-coded.
- Don't commit video, model weights, or datasets; use `.gitignore` and document where they live.
