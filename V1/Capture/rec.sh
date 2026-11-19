#!/usr/bin/env bash
# One-command recording on the Pi that keeps running if SSH drops (runs inside tmux).
#
#   ./V1/Capture/rec.sh start 20 fish     # record 20 minutes, label "fish"
#   ./V1/Capture/rec.sh status            # is it running? last lines + files so far
#   ./V1/Capture/rec.sh stop              # stop early (file is closed cleanly)
#   ./V1/Capture/rec.sh watch             # attach to the live output (detach: Ctrl+B then D)
#
# Needs: sudo apt install -y tmux   (plus record_usb.py's ffmpeg / v4l-utils / python3-yaml)
set -euo pipefail

SESSION=rec
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$REPO/recordings/rec.log"
cmd="${1:-help}"

running() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "$cmd" in
  start)
    minutes="${2:?usage: rec.sh start <minutes> [label]}"
    label="${3:-fish}"
    command -v tmux >/dev/null || { echo "tmux missing: sudo apt install -y tmux"; exit 1; }
    if running; then echo "Already recording. Use: rec.sh status | rec.sh stop"; exit 1; fi
    mkdir -p "$REPO/recordings"
    seconds=$(python3 -c "print(int(float('$minutes') * 60))")
    tmux new-session -d -s "$SESSION" -c "$REPO" \
      "python3 V1/Capture/record_usb.py --duration $seconds --label $label 2>&1 | tee -a '$LOG'"
    echo "Recording started: $minutes min, label '$label'."
    echo "Safe to close SSH. Check: ./V1/Capture/rec.sh status   Stop early: ./V1/Capture/rec.sh stop"
    ;;
  status)
    if running; then echo "● RECORDING"; else echo "○ not recording"; fi
    [ -f "$LOG" ] && { echo "--- last log lines ---"; tail -n 5 "$LOG"; }
    echo "--- newest files ---"
    ls -lht "$REPO/recordings" 2>/dev/null | grep -E '\.mkv|\.json' | head -n 4 || true
    df -h "$REPO" | tail -n 1 | awk '{print "--- free space: " $4 " ---"}'
    ;;
  stop)
    if running; then
      tmux send-keys -t "$SESSION" C-c
      for _ in $(seq 1 20); do running || break; sleep 0.5; done
      echo "Stopped."
    else
      echo "Not recording."
    fi
    ;;
  watch)
    running && tmux attach -t "$SESSION" || echo "Not recording."
    ;;
  *)
    sed -n 2,8p "$0"
    ;;
esac
