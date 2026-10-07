#!/usr/bin/env python3
"""Replay one RealSense recording through two static TensorRT engines."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.extend([ROOT, SCRIPTS_DIR])

from run_demo_single_trt import SingleEngineTrtRunner, normalize_imagenet


def engine_input_size(runner: SingleEngineTrtRunner) -> tuple[int, int]:
    """Return the static (width, height) required by a single-model engine."""
    shape = tuple(runner.engine.get_tensor_shape("left_image"))
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise RuntimeError(
            f"Expected a static [1, 3, H, W] left_image input, got {shape}"
        )
    if shape[2] <= 0 or shape[3] <= 0:
        raise RuntimeError("Dynamic-shape engines are not supported by this script")
    return shape[3], shape[2]


def disparity_to_depth_view(
    disparity: np.ndarray, fx: float, baseline: float, zfar: float
) -> np.ndarray:
    depth = np.full_like(disparity, np.nan, dtype=np.float32)
    valid = disparity > 0.1
    depth[valid] = fx * baseline / disparity[valid]
    normalized = np.nan_to_num(depth, nan=zfar, posinf=zfar)
    normalized = np.clip(normalized, 0.2, zfar)
    normalized = 1.0 - (normalized - 0.2) / (zfar - 0.2)
    return cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    cv2.putText(
        image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return image


def infer(
    runner: SingleEngineTrtRunner, left: torch.Tensor, right: torch.Tensor
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    outputs = runner({"left_image": left, "right_image": right})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    disparity = outputs["disparity"].squeeze().float().cpu().numpy()
    return disparity.clip(0, None), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two FFS TensorRT engines on the same RealSense .bag."
    )
    parser.add_argument("--bag", required=True, help="Input RealSense .bag file")
    parser.add_argument("--engine-a", required=True)
    parser.add_argument("--engine-b", required=True)
    parser.add_argument("--label-a", help="Display label for --engine-a")
    parser.add_argument("--label-b", help="Display label for --engine-b")
    parser.add_argument("--zfar", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--output-video", help="Optional MP4 path for the three-panel comparison"
    )
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--show", action="store_true", help="show a comparison window")
    args = parser.parse_args()

    bag = Path(args.bag)
    if not bag.is_file():
        parser.error(f"recording does not exist: {bag}")
    if args.zfar <= 0.2:
        parser.error("--zfar must be greater than 0.2")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if not args.show and not args.output_video:
        parser.error("choose --show, --output-video, or both")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    runner_a = SingleEngineTrtRunner(args.engine_a)
    runner_b = SingleEngineTrtRunner(args.engine_b)
    input_size_a = engine_input_size(runner_a)
    input_size_b = engine_input_size(runner_b)
    if input_size_a != input_size_b:
        raise RuntimeError(
            f"Engines need identical input dimensions, got {input_size_a} and "
            f"{input_size_b}"
        )
    input_width, input_height = input_size_a
    label_a = args.label_a or Path(args.engine_a).parent.name
    label_b = args.label_b or Path(args.engine_b).parent.name

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag.resolve()), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    writer = None

    try:
        left_profile = (
            profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        )
        right_profile = (
            profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        )
        intr = left_profile.get_intrinsics()
        extr = left_profile.get_extrinsics_to(right_profile)
        baseline = float(np.linalg.norm(extr.translation))
        if (intr.width, intr.height) != (input_width, input_height):
            raise RuntimeError(
                "The recording and both engines must use identical dimensions: "
                f"recording is {intr.width}x{intr.height}, engines require "
                f"{input_width}x{input_height}"
            )

        if args.output_video:
            output = Path(args.output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps,
                (input_width * 3, input_height), True,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output}")

        print(
            f"Comparing {label_a} and {label_b} on {bag} "
            f"({input_width}x{input_height}); press q or Esc to stop."
        )
        frame_count = 0
        total_a = 0.0
        total_b = 0.0
        while args.max_frames == 0 or frame_count < args.max_frames:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                raise

            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if not left_frame or not right_frame:
                continue
            left = np.asanyarray(left_frame.get_data())
            right = np.asanyarray(right_frame.get_data())
            left_rgb = np.repeat(left[..., None], 3, axis=2)
            right_rgb = np.repeat(right[..., None], 3, axis=2)
            left_t = torch.from_numpy(normalize_imagenet(left_rgb)).cuda()
            right_t = torch.from_numpy(normalize_imagenet(right_rgb)).cuda()
            left_t = left_t.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            right_t = right_t.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

            disparity_a, elapsed_a = infer(runner_a, left_t, right_t)
            disparity_b, elapsed_b = infer(runner_b, left_t, right_t)
            frame_count += 1
            total_a += elapsed_a
            total_b += elapsed_b

            left_view = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            view_a = disparity_to_depth_view(disparity_a, intr.fx, baseline, args.zfar)
            view_b = disparity_to_depth_view(disparity_b, intr.fx, baseline, args.zfar)
            add_label(left_view, "Input IR")
            add_label(view_a, f"{label_a}: {1000 * elapsed_a:.0f} ms")
            add_label(view_b, f"{label_b}: {1000 * elapsed_b:.0f} ms")
            comparison = np.hstack((left_view, view_a, view_b))

            if writer is not None:
                writer.write(comparison)
            if args.show:
                cv2.imshow("FFS TensorRT comparison", comparison)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_count:
        print(
            f"Compared {frame_count} frames. "
            f"{label_a}: {frame_count / total_a:.2f} FPS; "
            f"{label_b}: {frame_count / total_b:.2f} FPS."
        )
    else:
        print("No synchronized IR frames were read from the recording.")


if __name__ == "__main__":
    main()
