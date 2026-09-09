#!/usr/bin/env bash
# Launch Fast-FoundationStereo on a Jetson with the integrated GPU enabled.
#
# Optional environment variables:
#   IMAGE=ffs:jetson              container image to launch
#   NAME=ffs-jetson               container name
#   WITH_REALSENSE=1              expose USB and /dev/video* to the container
#   WITH_X11=1                    expose the local X11 display
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${IMAGE:-ffs:jetson}"
name="${NAME:-ffs-jetson}"

args=(
  --rm -it
  --name "$name"
  --runtime nvidia
  --network host
  --ipc host
  -v "$repo_root:/workspace"
  -w /workspace
)

if [[ "${WITH_REALSENSE:-0}" == "1" ]]; then
  # RealSense needs both USB control access (major 189) and UVC video nodes
  # (major 81). Add only present video devices instead of bind-mounting /dev.
  [[ -d /dev/bus/usb ]] && args+=(-v /dev/bus/usb:/dev/bus/usb)
  args+=(--device-cgroup-rule='c 81:* rmw' --device-cgroup-rule='c 189:* rmw')
  shopt -s nullglob
  for device in /dev/video*; do
    args+=(--device="$device")
  done
fi

if [[ "${WITH_X11:-0}" == "1" ]]; then
  : "${DISPLAY:?Set DISPLAY before using WITH_X11=1}"
  [[ -d /tmp/.X11-unix ]] || { echo 'Missing /tmp/.X11-unix on host' >&2; exit 1; }
  args+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

exec docker run "${args[@]}" "$image" bash
