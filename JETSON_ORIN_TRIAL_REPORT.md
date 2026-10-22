# Jetson Orin Fast-FoundationStereo trial report

## Current working setup

- **Hardware:** Jetson Orin NX 16 GB, JetPack 6.2.1 / L4T R36.4.4.
- **Container:** `ffs:jetson`, based on
  `dustynv/torch2trt:r36.4.0-cu128-24.04`.
- **Camera:** RealSense D456, IR stereo at **640x480 @ 30 FPS**; measured
  `fx=388.15 px`, baseline `0.094915 m`; IR emitter disabled.
- **Deployment path:** FP16 TensorRT engine, built locally on the Orin.

## What we resolved

| Area | Result |
|---|---|
| GPU support | Replaced a desktop PyTorch image that lacked Orin `sm_87` kernels. CUDA now runs on the Orin GPU. |
| TensorRT | Used the Jetson-native `torch2trt` image (PyTorch 2.6, TensorRT 10.7). The engine builder uses TensorRT's Python API because `trtexec` is absent. |
| Python dependencies | Installed OpenCV in the container virtual environment and avoided unsupported x86 TensorRT wheels. |
| RealSense | Built ARM64 `pyrealsense2` from librealsense with RSUSB, avoiding a Jetson kernel patch. |
| Camera access | The launcher exposes the D456 USB/UVC devices with `WITH_REALSENSE=1`. |
| X11 preview | Added the Qt/XCB dependencies, including `libsm6` and `libice6`. |

## Launching the Docker environment

Run these commands from the repository root **on the Jetson host**. Rebuild
after changing `docker/dockerfile.jetson` or its requirements:

```bash
docker build --network host -t ffs:jetson -f docker/dockerfile.jetson .
chmod +x docker/run_container_jetson.sh
```

Run the live camera pipeline without a preview window:

```bash
WITH_REALSENSE=1 bash docker/run_container_jetson.sh
```

For an OpenCV preview, run this once in the Jetson desktop session and then
launch the container:

```bash
xhost +si:localuser:root
WITH_REALSENSE=1 WITH_X11=1 bash docker/run_container_jetson.sh
```

The source repository is mounted at `/workspace`, so engines, recordings, and
comparison videos saved there persist on the Jetson host. Revoke X11 access
afterwards if desired:

```bash
xhost -si:localuser:root
```

## Engine parameter notation

Engine labels use `i<iterations>_d<max-disparity>`:

- **`i` / `valid_iters`:** recurrent disparity-refinement passes. Fewer
  passes are faster but can reduce stability at edges, occlusions, and weakly
  textured regions.
- **`d` / `max_disp`:** largest left/right pixel displacement searched by the
  model. Smaller values reduce cost-volume computation but limit near depth.
  With this D456 calibration, `d128`, `d96`, and `d64` correspond roughly to
  minimum depths of **0.29 m**, **0.38 m**, and **0.58 m**.

`max_disp` must be a positive multiple of **32** because of the model's
three-level 3D cost-aggregation hourglass. Thus `d64`, `d96`, and `d128` are
valid; `d48` is not.

## Inference results

All measurements use 640x480 camera input and `tegrastats`.

| Backend / engine | Power mode | Measured FPS | Notes |
|---|---:|---:|---|
| TensorRT, `i4_d128` | 25W | 1.5 | GPU was capped at 407 MHz. |
| PyTorch eager, `i4_d128` | 25W | 0.7 | CUDA works, but is not the deployment path. |
| TensorRT, `i4_d128` | MAXN | 2.9 | GPU reached ~912 MHz. |
| PyTorch eager, `i4_d128` | MAXN | 1.5 | Still substantially slower than TensorRT. |
| TensorRT, `i2_d96` | MAXN | 4.5 | Acceptable quality in testing. |
| TensorRT, `i2_d64` | MAXN | **5.0** | Current fastest tested acceptable configuration. |

The GPU is compute-bound, not thermally throttled: in MAXN it reached about
`GR3D_FREQ 98%@912`, with temperatures around 60°C. The 25W profile's 407 MHz
GPU cap explains the MAXN improvement.

## Recommended current configuration

Use **MAXN + TensorRT `i2_d64` at 640x480**, keeping `i4_d128` as the
quality-reference engine:

```bash
python3 scripts/live_realsense_trt.py \
  --engine output_ffs_i2_d64/fast_foundationstereo.engine \
  --width 640 --height 480 --fps 30 --zfar 10
```

Add `--show 0` for a headless FPS measurement.

## Building and testing a new engine version

Inside the container, give every experiment a unique output directory. For
example, build `i1_d64` at the unchanged 640x480 resolution:

```bash
python3 scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_ffs_i1_d64 \
  --height 480 --width 640 \
  --valid_iters 1 --max_disp 64

python3 scripts/build_single_trt_engine.py \
  --onnx output_ffs_i1_d64/fast_foundationstereo.onnx \
  --engine output_ffs_i1_d64/fast_foundationstereo.engine \
  --workspace-gb 2
```

Engine building is the slow step and took roughly 70 minutes in this setup.
Benchmark first without rendering:

```bash
python3 scripts/live_realsense_trt.py \
  --engine output_ffs_i1_d64/fast_foundationstereo.engine \
  --width 640 --height 480 --fps 30 --zfar 10 --show 0
```

Then omit `--show 0` to inspect live depth quality. Engines are tied to the
Jetson, TensorRT version, input dimensions, iteration count, and maximum
disparity used while building them.

## Visual comparison on the same camera sequence

Record synchronized D456 IR frames once as a RealSense `.bag` file. This is
better than using a compressed video because both engines receive the same
original left/right stereo frames and calibration:

```bash
python3 scripts/record_realsense_bag.py \
  --output recordings/scene_01.bag --seconds 15 \
  --width 640 --height 480 --fps 30
```

Replay that recording through the baseline and faster engines. The output has
three panels: input IR, `i4_d128` depth, and `i2_d64` depth:

```bash
python3 scripts/compare_realsense_trt.py \
  --bag recordings/scene_01.bag \
  --engine-a output_ffs_trt_jetson/fast_foundationstereo.engine \
  --label-a i4_d128 \
  --engine-b output_ffs_i2_d64/fast_foundationstereo.engine \
  --label-b i2_d64 \
  --zfar 10 --show \
  --output-video comparisons/i4_d128_vs_i2_d64.mp4
```

Recording needs `WITH_REALSENSE=1`. Replay does not require the physical
camera; `WITH_X11=1` is needed only for `--show`. Use `--max-frames N` to
make a shorter comparison clip.

## Trade-offs and next experiment

`d64` has an approximate closest usable depth of 0.58 m. No lower-resolution
configuration has been adopted. The next full-resolution experiment is
`i1_d64`; it may increase FPS but needs careful quality checks at boundaries
and in weakly textured areas.
