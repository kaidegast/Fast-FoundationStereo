# Fast-FoundationStereo on Jetson Orin GPU

This setup targets an ARM64 Jetson Orin running **JetPack 6.2.1 / L4T 36.4.4**.
It uses a Jetson-native PyTorch image with Orin (`sm_87`) CUDA kernels; the
repository's `docker/dockerfile` is for x86_64 PCs and must not be built on an
Orin.

The TensorRT engine in `output_ffs_trt/` was built elsewhere and is not
portable to Orin. Export ONNX and build a fresh engine on the Jetson.

## 1. Confirm the host is ready

On the Jetson host, check its JetPack/L4T release and install/configure Docker
and the NVIDIA Container Toolkit if necessary:

```bash
cat /etc/nv_tegra_release
docker info --format '{{.Runtimes}}'
```

The second command must list `nvidia`. If it does not (for example after
flashing with SDK Manager rather than the Jetson ISO), install/configure the
runtime on the **host** and start a new shell after the group change:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container curl
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl --now enable docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Confirm the runtime again with `docker info --format '{{.Runtimes}}'` before
building FFS. The launcher below selects `--runtime nvidia` explicitly.

For this JetPack 6 host, the default is
`dustynv/torch2trt:r36.4.0-cu128-24.04`. It provides the project's PyTorch
2.6 / torchvision 0.21 pair, TensorRT, and Orin-compatible CUDA kernels. The
available image tag is `r36.4.0`; it is compatible with the R36.4.x host
family. If the host uses a different JetPack release, choose a corresponding
Jetson-native image and pass it as `BASE_IMAGE` in the build command below.
Do not use ordinary desktop `nvidia/cuda` or `nvidia/cuda:12.4` images on
Jetson.

## 2. Build on the Jetson

Clone or copy this repository onto the Jetson, including the model checkpoint,
then run from the repository root:

```bash
docker build --network host -t ffs:jetson -f docker/dockerfile.jetson .
chmod +x docker/run_container_jetson.sh
```

The first build now also compiles the ARM64 `pyrealsense2` binding from
librealsense 2.56.5. It can take several minutes on the Orin; subsequent builds
reuse the Docker layer unless the RealSense version or base image changes. This
uses the RealSense `RSUSB` backend, so no host kernel patch is required.

For a different compatible base-image tag:

```bash
docker build --network host \
  --build-arg BASE_IMAGE=<JetPack-matched-Jetson-image> \
  -t ffs:jetson -f docker/dockerfile.jetson .
```

The image deliberately uses NVIDIA's CUDA-enabled PyTorch package. Installing
the x86 CUDA 12.4 PyPI wheels or the x86 Miniconda installer from the original
Dockerfile will fail on ARM64 or replace the Jetson-tuned stack.

OpenCV is installed with pip rather than Ubuntu's `python3-opencv` package:
the container's Python runs from its own virtual environment, which cannot
import Ubuntu system-Python packages.

## 3. Start a GPU container and verify it

```bash
bash docker/run_container_jetson.sh
```

Inside the container, confirm that PyTorch sees the integrated GPU:

```bash
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA:', torch.version.cuda)
try:
    import tensorrt as trt
    print('TensorRT:', trt.__version__)
except ImportError:
    print('TensorRT Python bindings unavailable')
PY
```

`nvidia-smi` is normally unavailable on Jetson; use `tegrastats` on the host
instead while inference is running.

The image also includes the RealSense binding. Verify it before starting the
live script:

```bash
python3 - <<'PY'
import pyrealsense2 as rs
print('pyrealsense2:', rs.__version__)
print('cameras:', [d.get_info(rs.camera_info.name) for d in rs.context().devices])
PY
```

## 4. First GPU inference

Use the PyTorch path first. Disabling point-cloud output avoids an optional
Open3D dependency that does not have reliable ARM64 wheels:

```bash
python3 scripts/run_demo.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output_jetson \
  --get_pc 0 --show 0 --valid_iters 4 --max_disp 128 --scale 0.5
```

The first invocation can take longer while CUDA kernels are prepared. Start
with the lower-resolution/4-iteration configuration, then raise resolution,
iterations, or `max_disp` only while memory and latency allow.

## 5. TensorRT deployment (recommended)

Build the engine on the Orin for the exact image size, model, TensorRT version,
and JetPack release that will run it. For a 640x480, four-iteration engine:

```bash
python3 scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_ffs_trt_jetson \
  --height 480 --width 640 --valid_iters 4 --max_disp 128

python3 scripts/build_single_trt_engine.py \
  --onnx output_ffs_trt_jetson/fast_foundationstereo.onnx \
  --engine output_ffs_trt_jetson/fast_foundationstereo.engine \
  --workspace-gb 2

python3 scripts/run_demo_single_trt.py \
  --model_file output_ffs_trt_jetson/fast_foundationstereo.engine \
  --model_dir output_ffs_trt_jetson \
  --left_file demo_data/left.png --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt --out_dir output_jetson_trt --get_pc 0 --show 0
```

The repository engine-builder uses TensorRT's Python API and does not require
`trtexec`, which is omitted from some otherwise complete Jetson TensorRT
containers. If TensorRT cannot import in the container, select a
JetPack-matched TensorRT image rather than mixing host Ubuntu packages or
unsupported desktop wheels into the container.

## RealSense or X11 (optional)

For a RealSense camera, start the container with the camera's USB/UVC devices:

```bash
WITH_REALSENSE=1 bash docker/run_container_jetson.sh
```

Then run the TensorRT live camera path (the engine must have been built for the
same `--width` and `--height`):

```bash
python3 scripts/live_realsense_trt.py \
  --engine output_ffs_trt_jetson/fast_foundationstereo.engine \
  --width 640 --height 480 --fps 30 --zfar 10 --show 0
```

`--show 0` is headless: it prints measured FPS every 30 frames and does not
need X11. For the preview window, first allow the container access to the local
X server and start it with `WITH_X11=1`, then omit `--show 0`. The Jetson image
includes the Qt/XCB runtime libraries required by OpenCV's preview window,
including the X11 session-management libraries (`libsm6` and `libice6`).

The launcher grants only the D456's USB/UVC device classes to the container.
If the import succeeds but `cameras: []` is printed, reconnect the camera to a
USB 3 port and relaunch with `WITH_REALSENSE=1`; do not install `pyrealsense2`
from desktop/x86 CUDA package indexes.

For OpenCV/Open3D windows, allow local-root access on the Jetson desktop once,
then launch with X11 mounted:

```bash
xhost +si:localuser:root
WITH_X11=1 bash docker/run_container_jetson.sh
```

Use both options together when live camera preview is needed. The launcher
does not mount all of `/dev` or `/home` into the container.
