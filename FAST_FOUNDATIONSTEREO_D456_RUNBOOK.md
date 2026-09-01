# Fast-FoundationStereo with Intel RealSense D456 (Docker)

This runbook reproduces the validated setup: Fast-FoundationStereo (FFS) running live on the left/right IR pair from an Intel RealSense D456, with the IR emitter disabled and TensorRT FP16 inference.

The validated host was Ubuntu 24.04.4 with an NVIDIA RTX 4050 Laptop GPU and a D456 connected at USB 3.2.

## Setup

### 1. Host prerequisites

Install and validate the following on the **host**, not inside the container:

- Docker Engine and NVIDIA Container Toolkit
- NVIDIA driver (`nvidia-smi` must work)
- Intel librealsense tools (`rs-enumerate-devices` must work)

Check the camera:

```bash
rs-enumerate-devices
```

The D456 should be listed and its `Usb Type Descriptor` should be USB 3.x.

### 2. Build and launch the FFS container

From the repository root:

```bash
docker build --network host -t ffs -f docker/dockerfile .
```

Edit `docker/run_container.sh` so the `docker run` command includes RealSense device access:

```bash
-v /dev:/dev \
--device-cgroup-rule='c 81:* rmw' \
--device-cgroup-rule='c 189:* rmw' \
```

Keep the existing `--gpus all` option. Launch it with:

```bash
bash docker/run_container.sh
```

Inside the container, verify GPU and video-device access:

```bash
nvidia-smi
ls -l /dev/video*
```

### 3. Install the validated Python environment

The repository expects PyTorch 2.6 with CUDA 12.4. Install the matched packages together:

```bash
python -m pip uninstall -y torch torchvision xformers

python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  xformers \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install gdown pyrealsense2
```

Verify:

```bash
python - <<'PY'
import torch, torchvision, xformers
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('xformers:', xformers.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
PY
```

If this is a disposable container, put these commands into the Dockerfile before rebuilding so they persist.

### 4. Download a checkpoint

Download checkpoint `23-36-37` (Already available in the repo):

```bash
mkdir -p weights/23-36-37

gdown 'https://drive.google.com/uc?export=download&id=1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj' \
  -O weights/23-36-37/cfg.yaml

gdown 'https://drive.google.com/uc?export=download&id=1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E' \
  -O weights/23-36-37/model_best_bp2_serialize.pth
```

Expected layout:

```text
weights/23-36-37/cfg.yaml
weights/23-36-37/model_best_bp2_serialize.pth
```

### 5. Validate FFS before using the camera

```bash
python scripts/run_demo.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output/ \
  --remove_invisible 0 \
  --denoise_cloud 1 \
  --scale 1 \
  --get_pc 1 \
  --valid_iters 4 \
  --max_disp 192 \
  --zfar 10
```

The first run can be much slower because CUDA/Triton kernels are compiled.

### 6. Validate the RealSense inside Docker

```bash
python - <<'PY'
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
print('devices:', len(devices))
for dev in devices:
    print(dev.get_info(rs.camera_info.name))
    print('serial:', dev.get_info(rs.camera_info.serial_number))
PY
```

The D456 should appear. FFS uses the hardware-synchronised `Infrared 1` and `Infrared 2` streams, not RGB plus IR.

## Test

### Capture and validate one D456 stereo pair

Use `848x480 @ 30 FPS` for the offline capture test. After `profile = pipeline.start(config)`, disable the emitter on every startup:

```python
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0)
print('IR emitter enabled:', depth_sensor.get_option(rs.option.emitter_enabled))
```

Run the following command from the repository root. It waits for auto-exposure to settle, then saves one hardware-synchronised `Infrared 1` / `Infrared 2` pair and the corresponding left-camera calibration:

```bash
python - <<'PY'
import os
import cv2
import numpy as np
import pyrealsense2 as rs

out_dir = 'realsense_data'
os.makedirs(out_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)

profile = pipeline.start(config)

try:
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print('IR emitter enabled:', depth_sensor.get_option(rs.option.emitter_enabled))

    # Discard initial frames so automatic exposure settles.
    for _ in range(30):
        frames = pipeline.wait_for_frames()

    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)
    left = np.asanyarray(left_frame.get_data())
    right = np.asanyarray(right_frame.get_data())

    left_profile = left_frame.profile.as_video_stream_profile()
    right_profile = right_frame.profile.as_video_stream_profile()
    intr = left_profile.get_intrinsics()
    extr = left_profile.get_extrinsics_to(right_profile)
    baseline = float(np.linalg.norm(extr.translation))

    cv2.imwrite(f'{out_dir}/left.png', left)
    cv2.imwrite(f'{out_dir}/right.png', right)

    K = [
        intr.fx, 0.0, intr.ppx,
        0.0, intr.fy, intr.ppy,
        0.0, 0.0, 1.0,
    ]
    with open(f'{out_dir}/K.txt', 'w') as f:
        f.write(' '.join(map(str, K)) + '\n')
        f.write(f'{baseline}\n')

    print(f'Saved pair to {out_dir}/')
    print(f'Resolution: {left.shape[1]}x{left.shape[0]}')
    print(f'fx={intr.fx:.3f}, fy={intr.fy:.3f}, baseline={baseline:.6f} m')
finally:
    pipeline.stop()
PY
```

This command creates:

```text
realsense_data/left.png
realsense_data/right.png
realsense_data/K.txt
```

`K.txt` has the flattened 3x3 left-IR intrinsic matrix on line 1 and the left-to-right baseline in metres on line 2. Run FFS on that pair:

```bash
python scripts/run_demo.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file realsense_data/left.png \
  --right_file realsense_data/right.png \
  --intrinsic_file realsense_data/K.txt \
  --out_dir output_realsense/ \
  --remove_invisible 1 \
  --denoise_cloud 1 \
  --get_pc 1 \
  --valid_iters 4 \
  --max_disp 192 \
  --zfar 10
```

## Run live

### PyTorch vs. TensorRT

Both runtimes execute the same trained FFS model and produce the same type of output: a disparity map that is converted to metric depth using the D456 focal length and stereo baseline. The difference is how the model is executed.

| PyTorch | TensorRT |
|---|---|
| Standard development runtime used by the original live script. | NVIDIA inference runtime that compiles the exported model into a GPU-specific engine. |
| Easy to inspect, modify, and debug. Supports variable camera resolutions through padding. | Optimised for deployment. The engine has a fixed input resolution and is tied to the TensorRT version and target GPU. |
| Executes the model as PyTorch CUDA operations, with less opportunity for whole-graph optimisation. | Selects GPU-specific kernels and fuses compatible operations; this runbook uses FP16 kernels. |
| Best for experimentation and debugging. | Best for the final live pipeline when latency and throughput matter. |

The FFS reference benchmarks report approximately a 2x speed-up from TensorRT over PyTorch for comparable configurations on an RTX 3090. Do not expect the same absolute FPS on an RTX 4050 Laptop GPU, but TensorRT is still the main performance lever after lowering resolution, refinement iterations, or maximum disparity.

### PyTorch

The repository-local script `scripts/live_realsense_ffs.py` runs the D456 live feed with PyTorch. It should use:

- IR 1 + IR 2 at the same resolution and frame rate
- `emitter_enabled = 0`
- three repeated channels for each monochrome IR image
- `torch.set_grad_enabled(False)`
- `torch.backends.cudnn.benchmark = True`
- `optimize_build_volume='triton'`

Launch at a lower fixed camera resolution first:

```bash
python scripts/live_realsense_ffs.py \
  --width 640 --height 480 --fps 30 \
  --iters 4 --max-disp 128
```

On the RTX 4050, the model-only PyTorch benchmark was about 131 ms per 640x480 stereo pair (~7.6 FPS), so TensorRT is the preferred deployment backend.

### TensorRT

#### Build the TensorRT engine

The FFS TensorRT export is static: the engine must be built for the exact camera resolution used at runtime. This runbook uses `640x480`.

Install TensorRT 10.16.1 for CUDA 12. TensorRT 11 has incompatible builder API changes for the repository's documented FP16 workflow.

```bash
python -m pip uninstall -y \
  tensorrt-cu12 tensorrt_cu12_libs tensorrt_cu12_bindings

python -m pip install --no-cache-dir 'tensorrt-cu12==10.16.1.11'
```

Export ONNX:

```bash
python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_ffs_trt \
  --height 480 --width 640 \
  --valid_iters 4 \
  --max_disp 128
```

Build the FP16 engine with TensorRT's Python API:

```bash
python - <<'PY'
from pathlib import Path
import tensorrt as trt

onnx_path = Path('output_ffs_trt/fast_foundationstereo.onnx')
engine_path = Path('output_ffs_trt/fast_foundationstereo.engine')

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError('ONNX parsing failed')

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError('TensorRT engine build failed')

engine_path.write_bytes(engine)
print(f'Saved {engine_path}')
PY
```

The engine build takes several minutes but is a one-time operation per GPU, TensorRT version, model, and resolution.

Validate it offline:

```bash
python scripts/run_demo_single_trt.py \
  --model_dir output_ffs_trt \
  --model_file output_ffs_trt/fast_foundationstereo.engine \
  --left_file realsense_data/left.png \
  --right_file realsense_data/right.png \
  --intrinsic_file realsense_data/K.txt \
  --out_dir output_realsense_trt \
  --get_pc 0 \
  --remove_invisible 1
```

#### Launch the live TensorRT pipeline

The repository-local script `scripts/live_realsense_trt.py` uses the TensorRT engine and applies the required ImageNet normalization before inference.

Launch it from the repository root:

```bash
python scripts/live_realsense_trt.py \
  --engine output_ffs_trt/fast_foundationstereo.engine \
  --width 640 --height 480 --fps 30 --zfar 10
```

Press `q` or `Esc` to exit. The window shows the left IR image and FFS metric depth. The IR emitter remains disabled.

## Notes and troubleshooting

- Do not swap left and right images.
- Use the RealSense IR pair, which is synchronised and calibrated by the camera.
- Keep calibration consistent with resolution. The live scripts read intrinsics and baseline from the active RealSense stream profile.
- The TensorRT engine is static: a 640x480 engine cannot be used with an 848x480 camera stream.
- Build TensorRT engines on the target GPU. Rebuild after a TensorRT major-version change.
- Connect the laptop to AC power and set the host to performance mode for best throughput:

  ```bash
  powerprofilesctl set performance
  ```

- To inspect GPU utilisation while running the live application, use another host terminal:

  ```bash
  watch -n 1 nvidia-smi
  ```
