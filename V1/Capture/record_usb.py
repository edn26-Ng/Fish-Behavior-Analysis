"""
Record the USB stereo camera on the Pi (Pi = capture only; processing is offline).

The camera sends ONE side-by-side MJPG frame (left | right). Frames are copied straight
into .mkv (no re-encode, so the Pi 4 keeps up at 3840x1080@30).

Per segment it writes:
  <stamp>_<label>_stereo_seg<K>.mkv        video, left half = left eye, right half = right eye
  <stamp>_<label>_stereo_seg<K>_ts.txt     one timestamp (ms from segment start) per frame
  <stamp>_<label>_stereo_seg<K>.json       settings, camera controls, stream start (epoch s), fps check

Frame wall time = json["stream_start_epoch_s"] + ts_ms / 1000.

Usage (on the Pi):
  python3 V1/Capture/record_usb.py --config V1/configs/capture_usb.yaml --duration 600 --label prefeed
  python3 V1/Capture/record_usb.py --show-controls      # list exposure/white-balance control names

Needs: sudo apt install -y ffmpeg v4l-utils python3-yaml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


def v4l2(device: str, *args: str) -> str:
    r = subprocess.run(["v4l2-ctl", "-d", device, *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def apply_controls(device: str, controls: dict) -> None:
    for k, v in (controls or {}).items():
        out = v4l2(device, f"--set-ctrl={k}={v}")
        if out:
            print(f"  ! {k}={v}: {out}")


def run_segment(cfg: dict, base: Path, seconds: float | None, stop_flag: dict, input_args: list[str]) -> dict:
    """Record one segment with ffmpeg. Returns stats for the JSON sidecar."""
    limit = ["-t", f"{seconds:.3f}"] if seconds else []  # output option: must be given per output
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-y",
        *input_args,
        *limit, "-map", "0:v", "-c", "copy", f"{base}.mkv",
        *limit, "-map", "0:v", "-c", "copy", "-f", "mkvtimestamp_v2", f"{base}_ts.txt",
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    stop_flag["proc"] = proc
    _, err = proc.communicate()
    stop_flag["proc"] = None
    elapsed = time.time() - t0

    if proc.returncode not in (0, 255):  # 255 = stopped with Ctrl-C (normal)
        print(err[-2000:])
        raise SystemExit(f"ffmpeg failed (exit {proc.returncode}). Check device/size/fps with --show-controls.")

    start = re.search(r"start: ([0-9.]+)", err)
    ts_lines = [l for l in Path(f"{base}_ts.txt").read_text().splitlines() if l and not l.startswith("#")]
    n = len(ts_lines)
    span_s = (float(ts_lines[-1]) - float(ts_lines[0])) / 1000 if n > 1 else 0.0
    return {
        "frames": n,
        "measured_fps": round((n - 1) / span_s, 2) if span_s else None,
        "elapsed_s": round(elapsed, 2),
        "stream_start_epoch_s": float(start.group(1)) if start else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the USB stereo camera for offline processing.")
    ap.add_argument("--config", default="V1/configs/capture_usb.yaml")
    ap.add_argument("--duration", type=float, help="seconds; 0 = until Ctrl-C (overrides config)")
    ap.add_argument("--label", help="filename label (overrides config)")
    ap.add_argument("--show-controls", action="store_true", help="list camera controls and exit")
    ap.add_argument("--input", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)  # testing: raw ffmpeg input args
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dev, v, rec = cfg["device"], cfg["video"], cfg["recording"]

    if args.show_controls:
        print(v4l2(dev, "--list-ctrls-menus"))
        return

    if args.duration is not None:
        rec["duration_s"] = args.duration or None
    if args.label:
        rec["label"] = args.label

    input_args = args.input or [
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{v['width']}x{v['height']}", "-framerate", str(v["fps"]),
        "-timestamps", "abs",  # kernel frame timestamps on the wall clock → stream_start_epoch_s
        "-i", dev,
    ]

    out = Path(rec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    stop = {"set": False, "proc": None}

    def on_signal(*_):
        stop["set"] = True
        if stop["proc"] and stop["proc"].poll() is None:
            stop["proc"].send_signal(signal.SIGINT)  # lets ffmpeg close the .mkv cleanly

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not args.input:
        apply_controls(dev, cfg.get("controls"))
        controls_now = v4l2(dev, "--list-ctrls")
    else:
        controls_now = "test input"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_end = time.monotonic() + rec["duration_s"] if rec["duration_s"] else None
    seg = 0
    while not stop["set"]:
        if shutil.disk_usage(out).free / 1e9 < rec["min_free_gb"]:
            print(f"Stopping: under {rec['min_free_gb']} GB free.")
            break
        seconds = rec["segment_s"]
        if t_end:
            remaining = t_end - time.monotonic()
            if remaining <= 0.5:
                break
            seconds = min(seconds, remaining)

        base = out / f"{stamp}_{rec['label']}_stereo_seg{seg:03d}"
        started = datetime.now().isoformat(timespec="milliseconds")
        print(f"Recording segment {seg} → {base.name}.mkv")
        stats = run_segment(cfg, base, seconds, stop, input_args)
        meta = {
            "device": dev,
            "layout": "side_by_side_left_right",
            "width": v["width"], "height": v["height"],
            "eye_width": v["width"] // 2, "fps_requested": v["fps"],
            "started": started, "segment": seg,
            **stats,
            "controls_set": cfg.get("controls") or {},
            "controls_now": controls_now,
        }
        Path(f"{base}.json").write_text(json.dumps(meta, indent=2))
        print(f"  {stats['frames']} frames, {stats['measured_fps']} fps measured")
        if stats["measured_fps"] and stats["measured_fps"] < 0.9 * v["fps"]:
            print("  ! Dropping frames: try 2560x720 or a faster SD card / USB SSD.")
        seg += 1
    print(f"Done. Files in {out.resolve()}")


if __name__ == "__main__":
    main()
