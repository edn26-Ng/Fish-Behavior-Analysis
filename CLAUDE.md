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
- **Camera (current): USB stereo "3D USB Camera"** (USB ID `32e4:0035`): one side-by-side MJPG frame (left | right), up to **3840×1080 @ 60 fps = 1920×1080 per eye**. Replaced the ribbon modules (Nov 2026).
- Planned mount: top-down, ~69 cm (27") above water (water surface is ~29" below the tank rim), ~10 cm baseline. Short image axis covers ~91 cm of the ~125 cm tank → ~84% of the surface is in frame; two thin edge strips are cut off.
- **Pi:** Raspberry Pi 4 (hostname `Rpifishfeeder`), Raspberry Pi OS 64-bit, controlled over **SSH from a laptop on a phone hotspot** (no monitor/keyboard). Find its IP: `1..14 | % { ping -n 2 -w 1000 "172.20.10.$_" | Out-Null }; arp -a` → MAC starting `dc-a6-32`/`2c-cf-67`/`d8-3a-dd`/`e4-5f-01`.
- ⚠️ The first Pi 4 board has a **dead USB controller** (`lsusb` empty, `dmesg`: `brcm-pcie … link down`, persists after bootloader update) → don't use it for the camera. Always check `lsusb` shows the **VIA Labs hub** first.
- Night recording open: candidate 940 nm IR + NoIR cameras.

## USB stereo capture (`V1/Capture/record_usb.py`) 🟢 current
Records the USB stereo camera on the Pi; processing is offline.
- **Install (Pi):** `sudo apt install -y ffmpeg v4l-utils python3-yaml git`
- **Check camera:** `lsusb` → `3D USB Camera`; `v4l2-ctl --list-devices` → first node (usually `/dev/video0`) goes in `capture_usb.yaml`.
- **Record:** `python3 V1/Capture/record_usb.py --config V1/configs/capture_usb.yaml --duration 600 --label prefeed` (`--duration 0` = until Ctrl-C)
- **Camera controls:** `--show-controls` lists names; set fixed exposure/white balance under `controls:` once tank lighting is known.
- **Output per segment** (`recordings/`, gitignored):
  - `…_stereo_seg<K>.mkv`: MJPG copied from the camera, **left half = left eye**
  - `…_ts.txt`: one timestamp per frame (ms from segment start, mkv timecode v2)
  - `…json`: settings, controls, `stream_start_epoch_s`, measured fps. **Frame wall time = `stream_start_epoch_s` + ts/1000**
- **Download (from laptop):** `scp nk693@<pi-ip>:~/Fish-Behavior-Analysis/recordings/* .`
- **Verified on the Pi 4:** `record_usb.py` 60 s at 3840×1080 @ 30 → **1797 frames, 29.94 fps, 654 MB** ✅
- ⚠️ **Storage: ~39 GB/hour at 3840×1080@30** (654 MB/min measured). 64 GB card (48 GB free after OS) holds ~1 h 10 min → **USB SSD needed for multi-hour pre-feed recording**, or drop to 10 fps / 2560×720 for slow pre-feed periods.
- **Gotchas:** stopping with Ctrl-C closes the file cleanly (ffmpeg gets SIGINT). Frame drops are flagged if measured fps < 90% of requested. Download over the hotspot runs ~11 MB/s (≈1 h per hour of footage) → move long recordings with a USB SSD/stick.
- **Long runs over SSH:** run inside `tmux` (`tmux new -s rec`, detach Ctrl+B D, `tmux attach -t rec`); a dropped SSH session otherwise kills the recording.

## Stereo calibration (`V1/Tools/calibrate_stereo.py`) 🟡 needs first real run
Gets each eye's lens distortion, the baseline and the rectification needed for depth. Run on the **laptop**.
- **Board:** `docs/calibration/checkerboard_letter_25mm.pdf` (10×7 squares = **9×6 inner corners**, 25 mm). Print at 100% / Actual size, **measure a square**, glue flat, keep the white border. 11×17 version (35 mm) also there. Never tile sheets.
- **Record (Pi, camera still):** `python3 V1/Capture/record_usb.py --duration 90 --label calib` while moving the board slowly through all **9 image zones**, ~50/75/100 cm, with tilts, pausing ~1 s per pose, fully inside **both** eyes.
- **Run:** `python V1/Tools/calibrate_stereo.py --video <…calib…seg000.mkv> --square-mm <measured>`
- **Output:** `V1/configs/stereo_calib.yaml` (K/D per eye, R/T, R1 R2 P1 P2 Q, `baseline_mm`, RMS, 3×3 coverage) + `stereo_calib_check.jpg` (rectified pair with horizontal lines, gitignored).
- **Pass:** stereo RMS < ~1 px · baseline ≈ ruler-measured lens gap · no empty coverage zones in either eye. Commit the YAML once it passes.
- **Gotchas:** calibration is **in air**; underwater depth needs a refraction correction (bucket test still to do). Recalibrate if the camera is dropped or anything on it changes. Uses OpenCV's rational distortion model (wide lenses). Verified on synthetic stereo video: recovered a 60.0 mm baseline exactly.

## Ribbon-camera capture (`V1/Capture/record.py`) ⚪ not in use
For the Camera Module 3 (picamera2) setup that was replaced by the USB stereo camera. Kept in case CSI cameras return.
Records on the Pi; all processing happens offline on a laptop.
- **Install (Pi):** `sudo apt install -y python3-picamera2 python3-yaml ffmpeg git`. picamera2 is apt-only; a venv needs `--system-site-packages`.
- **Cable:** Pi 4 uses the standard **15-pin** ribbon (ships with the camera, 200 mm). The 22-pin "Pi 5" cables do **not** fit a Pi 4. A 27" mount needs a longer 15-pin cable or the Pi mounted on the frame.
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
