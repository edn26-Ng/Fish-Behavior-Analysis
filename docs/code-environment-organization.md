# Code Environment Organization — Fish Behavior Analysis

## Chat: Detection & Tracking Pipeline Plan (FishNet evaluation → offline tracker CLI)
*Planning session, Nov 2026. Supersedes the "Current task: webcam demo" section of the previous CLAUDE.md.*

---

## How to respond
- Answer immediately, no preamble or restating the question.
- State conclusions directly; don't narrate alternatives.
- No filler or hedging. Minimum tokens that stay correct.
- Code or direct answer first; explain only non-obvious details.
- Optimize for ADHD: short sections, bullets, tables, bold key points.

## Project in one paragraph
Hand-labeled video library of tilapia group behavior → train a model that classifies group state (Resting / Hungry / Feeding / Satiated, plus a Disturbance event tag) → later drive an automated feeder that minimizes feed waste. Pi = capture only. Full spec: `docs/project-description.md`; literature + budget: `docs/research-notes.md`.

---

## Decisions made in this chat

| Topic | Decision |
|---|---|
| **FishNet** | **Not a detector.** It is a re-identification recipe (paper only: no released code, weights or dataset). It takes *already-cropped* fish and outputs a 128-number "fingerprint" to tell individuals apart. **Deferred**: build only if measured ID switches corrupt the data. |
| **What we reuse from FishNet** | Its **data bootstrap**: hand-label ~100 frames → train → let the model pre-label → correct → repeat to ~300–500 frames. |
| **Detector** | **YOLOv8n fine-tuned on tilapia.** Stock COCO `yolov8n.pt` (no fish class) is used until the tilapia weights exist. Swapped via the model path in config. |
| **Tracker** | **Configured and tested, not trained.** Ultralytics BoT-SORT vs ByteTrack, compare on the same clips. Camera is fixed → camera-motion compensation off. Re-ID off. |
| **Processing mode** | **Offline, every frame, no drops.** Record first, process later. Timestamps come from the video file, not the wall clock. |
| **"trackerCLI"** | = `tools/track_video.py` (video in → tracks CSV out). |
| **`demo/`** | Kept as the webcam concept demo. Not extended. New work goes in `pipeline/` and `tools/`. |
| **Current scope** | Capture, Preprocess, Detect, Track, Log, plus tools to verify and label. Features, classification and Pi capture come later. |

---

## Architecture

```
Capture → Preprocess → Detect → Track → Log        (this phase)
                                    └→ Extract features → Classify → Feeder   (later)
```

Each stage is a **separate, swappable module** with fixed input/output. Swapping webcam ↔ file ↔ Pi, or COCO ↔ tilapia model, is a config change only.

| Stage | Input | Output |
|---|---|---|
| Source | Video file / webcam / Pi | `Frame(image, index, timestamp)` |
| Preprocess | Raw frame | Cleaned frame (+ crop offset) |
| Detect | Cleaned frame | Boxes: bbox, conf, class |
| Track | Boxes, every frame in order | Boxes + stable `track_id` |
| Log | Tracked boxes + frame info | Per-frame CSV |

---

## Repo layout

**Status key:** ✅ **Implement now** (testable on any video with stock YOLOv8n) · 🦴 **Skeleton**: create the file with function signatures, docstrings and `NotImplementedError` / TODOs; fill in once tilapia recordings arrive.

```
Fish-Behavior-Analysis/
├── demo/                       # webcam concept demo (unchanged)
├── pipeline/
│   ├── source.py               ✅ frame sources
│   ├── preprocess.py           🦴 mask/crop/undistort (needs tank footage + calibration)
│   ├── detector.py             ✅ YOLO wrapper, model path from config
│   ├── tracker.py              ✅ tracking on every frame
│   └── logger.py               ✅ per-frame CSV writer
├── tools/
│   ├── track_video.py          ✅ main CLI ("trackerCLI")
│   ├── replay.py               ✅ draw CSV back onto video
│   ├── track_report.py         🦴 tracking-quality report (thresholds need fish data)
│   └── extract_frames.py       🦴 frames for labeling (needs tilapia recordings)
├── configs/
│   ├── pipeline.yaml           ✅ source, preprocess, model path, tracker, output
│   └── tracker_fish.yaml       🦴 copy of Ultralytics botsort.yaml; tune on fish video
├── datasets/tilapia.yaml       🦴 dataset config for fine-tuning (gitignored data)
└── models/                     # gitignored; weights live here (document location in README)
```

---

## File purposes and implementation

### `pipeline/source.py` ✅
**Job:** turn any input into the same stream of frames.
- `Frame` dataclass: `image` (BGR), `index` (0-based), `timestamp` (seconds).
- `BaseSource`: `read() -> Frame | None`, `release()`, iteration (`for frame in src`), context manager.
- `FileSource(path, fps_override=None)`: **main source.** `timestamp = index / fps`, so processing speed never affects measured fish speed. Raises if no valid FPS.
  - `fps_override` is **required for raw Pi `.h264`** files, which store no frame rate (or convert to `.mp4` first).
- `WebcamSource`: quick tests only. `CAP_DSHOW` on Windows, `CAP_ANY` elsewhere (the demo's DirectShow-only code fails on Linux/Pi). Timestamps = elapsed `perf_counter()`.
- `PicameraSource`: raises `NotImplementedError` until the **Pi 4 + HAT vs Pi 5** decision.
- `build_source(cfg)`: picks the source from config. Add new sources here only.
- Later: `StereoFileSource` returning matched left/right frame pairs.

### `pipeline/preprocess.py` 🦴
**Job:** frame in → cleaned frame out, *before* the detector. Each step is toggled in config. Skeleton now: a pass-through function with config hooks.

| Order | Step | When | Notes |
|---|---|---|---|
| 1 | Undistort / rectify | After calibration | Build maps once (`cv2.initUndistortRectifyMap`), apply per frame (`cv2.remap`). **Calibrate through water**, not air. |
| 2 | Tank mask | After first footage | Circle (center + radius) set once per camera position, stored in config. Blacks out rim, people, equipment. |
| 3 | Crop to mask | After first footage | Return the **crop offset** so boxes can be logged in original-frame pixels. |
| 4 | CLAHE brightness | Optional | Enable only if testing shows it helps. |
| 5 | Denoise | Night/IR | Off by default (slow). |

- **No manual resizing for YOLO**: it scales to `imgsz` internally.
- **Training and inference must use identical preprocessing** (`extract_frames.py` calls this same module).

### `pipeline/detector.py` ✅
**Job:** load weights from the **model path** and return boxes for one frame. No memory between frames.
- The model path = location of the `.pt` weights file. Architecture (YOLOv8n) stays the same; **only the weights change** what is detected.
- `yolov8n.pt` (auto-downloads, COCO) now → `models/tilapia_v1.pt` later.
- Returns a list of `{bbox: [x1,y1,x2,y2], conf, cls_id, cls_name}`.
- Owns the YOLO object; `tracker.py` uses the same object.

### `pipeline/tracker.py` ✅
**Job:** link boxes across frames into stable IDs.
- Calls `model.track(frame, persist=True, tracker=<config path>)` on **every frame, sequentially**. Never drop frames (the demo's `ThreadedDetector` drops them, which causes ID switches and fake speeds).
- Tracker choice and settings come from `configs/tracker_fish.yaml`.
- Returns detector dicts + `track_id`.

### `pipeline/logger.py` ✅
**Job:** write one CSV row per tracked object per frame.
- Columns: `video, frame, timestamp_s, track_id, x1, y1, x2, y2, cx, cy, conf, cls`.
- Coordinates in **original-frame pixels** (crop offset added back).
- Save a copy of the run's config next to the CSV for reproducibility.

### `tools/track_video.py` ✅ (main program / "trackerCLI")
**Job:** source → preprocess → detect + track → log, for one video.
```bash
python tools/track_video.py --config configs/pipeline.yaml --video recordings/clip1.mp4 --out runs/tracks/clip1.csv
```
- CLI args override config values. Progress printout; no display window needed.

### `tools/replay.py` ✅
**Job:** draw the CSV's boxes and IDs back onto the video. Stage-5 check that **replay matches the CSV**. Option to save the overlay video.

### `tools/track_report.py` 🦴
**Job:** measure tracking quality. Decides whether re-ID (FishNet recipe) is ever needed.
- Number of unique IDs vs estimated fish count (50–100), track-length distribution, fragmented tracks, impossible jumps (likely ID switches).
- Skeleton now; jump/length thresholds need real fish data.

### `tools/extract_frames.py` 🦴
**Job:** pull labeling frames **spread out in time** (e.g. every 3–5 s), skip near-duplicates, apply the same preprocessing, write images for CVAT/Roboflow.

### `configs/pipeline.yaml` ✅
```yaml
source:
  type: file                  # file | webcam | picamera2
  path: recordings/clip1.mp4
  fps_override: null          # e.g. 30 for raw Pi .h264
preprocess:
  undistort: false
  calibration_file: null
  tank_mask: {enabled: false, center: null, radius: null}
  crop_to_mask: false
  clahe: false
  denoise: false
detector:
  model: yolov8n.pt           # later: models/tilapia_v1.pt
  conf: 0.25
  classes: null               # testing: [0] = person
tracker: configs/tracker_fish.yaml
output:
  dir: runs/tracks
```

### `configs/tracker_fish.yaml` 🦴
Copy of Ultralytics `botsort.yaml`; tune once fish video exists. **Change one setting at a time.**

| Setting | Meaning | Fish note |
|---|---|---|
| `tracker_type` | botsort / bytetrack | Compare both; keep fewer ID switches |
| `track_high_thresh` | Confidence for confident matches | Too high → clumped fish dropped |
| `new_track_thresh` | Confidence to start a new ID | Too low → glare/ripples get IDs |
| `track_buffer` | Frames a lost fish is remembered | Raise for fish passing under others; scale with FPS |
| `match_thresh` | Box-matching strictness | Fish turn fast; may need loosening |
| `gmc_method` | Camera-motion compensation | **`none`**: camera is fixed |
| `with_reid` | Appearance re-ID | **Off** until ID switches are proven a problem |

---

## Fine-tuning the tilapia detector (once recordings exist)
1. **Record varied clips**: rest, feeding, clumps, glare, splashing, lab traffic, different lighting and days.
2. **Extract** spaced-out frames (`extract_frames.py`).
3. **Label** every visible fish, one class `fish`, with written rules: include partial and edge fish, tight boxes including the tail, separate boxes for overlapping fish. *An unlabeled fish teaches "not a fish."*
4. **Export YOLO format**: `images/{train,val}`, `labels/{train,val}`, one `class cx cy w h` line per fish (normalized). **Split by clip, not by frame.**
5. **Train** (GPU / Colab / Roboflow, never the Pi):
   ```bash
   yolo detect train data=datasets/tilapia.yaml model=yolov8n.pt epochs=100 imgsz=640 flipud=0.5
   ```
   Raise `imgsz` (960–1280) if fish look tiny. `flipud` is valid because the view is top-down.
6. **Evaluate**: precision, recall, mAP50, **plus watching a never-seen clip** (clumps, glare, edges).
7. **Model-assisted loop**: ~100 hand labels → v1 pre-labels → correct → retrain. Prioritize frames the model gets wrong.
8. **Deploy**: copy `runs/detect/train/weights/best.pt` → `models/tilapia_v1.pt` (versioned, gitignored), note the location in the README, update `detector.model`.

## Tracker evaluation
- **Automatic report** (`track_report.py`): ID count vs fish count, track lengths, jumps.
- **Visual** (`replay.py`): do IDs survive crossings?
- **Ground truth** (later): hand-label IDs on a 30–60 s clip (CVAT/DarkLabel), score IDF1.
- **Cleanup before features**: drop tracks under ~1 s, flag impossible jumps, smooth (Savitzky–Golay) before differentiating.

---

## Deferred / blocked

| Item | Waiting on |
|---|---|
| Tilapia weights | Tank recordings + ~300–500 labeled frames |
| Preprocess mask/undistort values | Tank footage + checkerboard calibration in water |
| `PicameraSource` / Pi recording script | **Pi 4 + stereo HAT vs Pi 5** decision |
| Stereo depth / `StereoFileSource` | Cameras mounted + calibrated |
| Feature extraction (speed, accel, dispersion, Delaunay, depth) | Working tracks |
| Behavior classifier (temporal window) | Features + behavior labels |
| FishNet-style re-ID | `track_report.py` showing harmful ID switches |

## Known issues in `demo/` (not fixing; don't copy into `pipeline/`)
- `ThreadedDetector` drops frames, and results lag the displayed frame.
- `CAP_DSHOW` is hard-coded (Windows only).
- `build_source` supports webcam only; the config lists file/picamera2.
- Default model is `yolov8s`, not `yolov8n`.
- Hand `track_id = idx` is a list position, not a real track ID.
- `claude/project-description.md` is a stale duplicate of `docs/project-description.md` (60 cm vs 69 cm camera height).

## Conventions
- Feature branches; PRs into `main`.
- All tunable values in YAML config or CLI args, never hard-coded.
- Never commit video, weights or datasets (`.gitignore`); document where they live.
- Never overwrite raw video. Preprocessing happens in memory.
- Keep full-frame-rate video; labels are sparse (~every 3 s), but kinematics need consecutive frames.
- Speed is px/frame → convert with FPS and in-water calibration (cm). Top-down pixel scale changes with depth.
