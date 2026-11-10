"""
Record video on the Raspberry Pi with Picamera2 (Pi = capture only; processing is offline).

Per clip, per camera it writes:
  <stamp>_<label>_cam<N>_seg<K>.mp4          H.264 video
  <stamp>_<label>_cam<N>_seg<K>_frames.csv   one row per encoded frame: frame, sensor_ts_us, wall_time
  <stamp>_<label>_cam<N>_seg<K>.json         settings + locked exposure/gain/white balance

sensor_ts_us lets stereo left/right frames be matched later (cameras are not hardware-synced).

Usage (on the Pi):
  python3 V1/Capture/record.py --config V1/configs/capture.yaml
  python3 V1/Capture/record.py --config V1/configs/capture.yaml --duration 600 --label prefeed

Install: picamera2 comes from apt, not pip:
  sudo apt install -y python3-picamera2 ffmpeg
  (venv must be created with --system-site-packages to see it)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
except ImportError as e:  # not on a Pi
    raise SystemExit(
        "picamera2 not found. Run this on the Pi after: sudo apt install -y python3-picamera2 ffmpeg"
    ) from e


class TimestampedFfmpegOutput(FfmpegOutput):
    """FfmpegOutput that also logs the sensor timestamp of every frame actually written to the video.

    Logging here (not in a camera callback) guarantees CSV rows == video frames.
    """

    def __init__(self, path: str, csv_path: str):
        super().__init__(path)
        self._csv_file = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame", "sensor_ts_us", "wall_time"])
        self.frames = 0

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        written = self.recording and not self.output_broken
        super().outputframe(frame, keyframe, timestamp, packet, audio)
        if written and not self.output_broken:  # only log frames that reached the video file
            self._csv.writerow([self.frames, timestamp, f"{time.time():.6f}"])
            self.frames += 1

    def stop(self):
        super().stop()
        self._csv_file.close()


class CameraRecorder:
    """One camera: configure, lock exposure, record segments, log per-frame timestamps."""

    def __init__(self, cam_num: int, cfg: dict):
        self.cam_num = cam_num
        self.cfg = cfg
        v = cfg["video"]
        self.picam = Picamera2(camera_num=cam_num)
        frame_us = int(1_000_000 / v["fps"])
        config = self.picam.create_video_configuration(
            main={"size": (v["width"], v["height"])},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        )
        self.picam.configure(config)
        self.locked_controls: dict = {}
        self.output: TimestampedFfmpegOutput | None = None

    def start(self) -> None:
        self.picam.start()
        ctrl = self.picam.camera_controls
        if "AfMode" in ctrl and self.cfg["focus"].get("lens_position") is not None:
            # Fixed focus so autofocus can't hunt on moving fish/ripples.
            self.picam.set_controls({"AfMode": 0, "LensPosition": float(self.cfg["focus"]["lens_position"])})

        exp = self.cfg["exposure"]
        time.sleep(exp["warmup_s"])
        if exp["lock"]:
            md = self.picam.capture_metadata()
            controls = {
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": int(exp["exposure_us"] or md["ExposureTime"]),
                "AnalogueGain": float(exp["analogue_gain"] or md["AnalogueGain"]),
            }
            if "ColourGains" in md:
                controls["ColourGains"] = tuple(md["ColourGains"])
            self.picam.set_controls(controls)
            self.locked_controls = {k: v for k, v in controls.items()}
            time.sleep(0.5)  # let locked values take effect

    def start_segment(self, base: Path) -> None:
        v = self.cfg["video"]
        self.encoder = H264Encoder(bitrate=int(v["bitrate"]))
        self.output = TimestampedFfmpegOutput(f"{base}.mp4", f"{base}_frames.csv")
        self.picam.start_encoder(self.encoder, self.output)
        meta = {
            "camera_num": self.cam_num,
            "sensor": self.picam.camera_properties.get("Model"),
            "start_time": datetime.now().isoformat(timespec="milliseconds"),
            "width": v["width"],
            "height": v["height"],
            "fps": v["fps"],
            "bitrate": v["bitrate"],
            "locked_controls": self.locked_controls,
            "lens_position": self.cfg["focus"].get("lens_position"),
        }
        Path(f"{base}.json").write_text(json.dumps(meta, indent=2, default=list))

    def stop_segment(self) -> int:
        self.picam.stop_encoder()  # also stops the output, which closes the CSV
        return self.output.frames if self.output else 0

    def close(self) -> None:
        self.picam.stop()
        self.picam.close()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description="Record Pi camera video for offline processing.")
    ap.add_argument("--config", default="V1/configs/capture.yaml")
    ap.add_argument("--duration", type=float, help="seconds (overrides recording.duration_s; 0 = until Ctrl-C)")
    ap.add_argument("--label", help="filename label (overrides recording.label)")
    ap.add_argument("--cameras", type=int, nargs="+", help="camera numbers, e.g. 0 1")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rec = cfg["recording"]
    if args.duration is not None:
        rec["duration_s"] = args.duration or None
    if args.label:
        rec["label"] = args.label
    if args.cameras:
        cfg["cameras"] = args.cameras

    out = Path(rec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    cams = [CameraRecorder(n, cfg) for n in cfg["cameras"]]
    for c in cams:
        c.start()
    print(f"Cameras {cfg['cameras']} ready. Locked: {[c.locked_controls for c in cams]}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_end = time.monotonic() + rec["duration_s"] if rec["duration_s"] else None
    seg = 0
    try:
        while not stop.is_set():
            if free_gb(out) < rec["min_free_gb"]:
                print(f"Stopping: under {rec['min_free_gb']} GB free.")
                break
            bases = [out / f"{stamp}_{rec['label']}_cam{c.cam_num}_seg{seg:03d}" for c in cams]
            for c, b in zip(cams, bases):
                c.start_segment(b)
            print(f"Recording segment {seg} → {bases[0].name}…")

            seg_end = time.monotonic() + rec["segment_s"]
            if t_end:
                seg_end = min(seg_end, t_end)
            while not stop.is_set() and time.monotonic() < seg_end:
                time.sleep(0.2)

            for c, b in zip(cams, bases):
                n = c.stop_segment()
                print(f"  cam{c.cam_num}: {n} frames → {b.name}.mp4")
            seg += 1
            if t_end and time.monotonic() >= t_end:
                break
    finally:
        for c in cams:
            c.close()
    print(f"Done. Files in {out.resolve()}")


if __name__ == "__main__":
    main()
