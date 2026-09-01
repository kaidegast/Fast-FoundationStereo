import os
import sys
import time
import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--zfar", type=float, default=10.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.set_grad_enabled(False)
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps
    )
    config.enable_stream(
        rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps
    )

    profile = pipeline.start(config)

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print("IR emitter enabled:", depth_sensor.get_option(rs.option.emitter_enabled))

        # Allow auto-exposure to settle and obtain camera calibration.
        for _ in range(30):
            frames = pipeline.wait_for_frames()

        left_profile = (
            frames.get_infrared_frame(1).profile.as_video_stream_profile()
        )
        right_profile = (
            frames.get_infrared_frame(2).profile.as_video_stream_profile()
        )
        intr = left_profile.get_intrinsics()
        extr = left_profile.get_extrinsics_to(right_profile)
        baseline = float(np.linalg.norm(extr.translation))

        print(
            f"Live FFS: {args.width}x{args.height} @ {args.fps} FPS, "
            f"fx={intr.fx:.2f}, baseline={baseline:.6f} m"
        )
        print("Press q or Esc to stop.")

        while True:
            t0 = time.perf_counter()

            frames = pipeline.wait_for_frames()
            left = np.asanyarray(frames.get_infrared_frame(1).get_data())
            right = np.asanyarray(frames.get_infrared_frame(2).get_data())

            # FFS accepts 3-channel images; duplicate the monochrome IR channel.
            left_rgb = np.repeat(left[..., None], 3, axis=2)
            right_rgb = np.repeat(right[..., None], 3, axis=2)

            left_t = (
                torch.from_numpy(left_rgb)
                .cuda()
                .float()
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            right_t = (
                torch.from_numpy(right_rgb)
                .cuda()
                .float()
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )

            padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
            left_t, right_t = padder.pad(left_t, right_t)

            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                disparity = model.forward(
                    left_t,
                    right_t,
                    iters=args.iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )

            disparity = padder.unpad(disparity.float())
            disparity = disparity.squeeze().detach().cpu().numpy().clip(0, None)

            # depth [m] = focal length [px] × baseline [m] / disparity [px]
            depth = np.full_like(disparity, np.nan, dtype=np.float32)
            valid = disparity > 0.1
            depth[valid] = intr.fx * baseline / disparity[valid]

            # Colourise depth: near = warm, far = cool.
            depth_vis = np.nan_to_num(depth, nan=args.zfar, posinf=args.zfar)
            depth_vis = np.clip(depth_vis, 0.2, args.zfar)
            depth_vis = 1.0 - (depth_vis - 0.2) / (args.zfar - 0.2)
            depth_vis = cv2.applyColorMap(
                (depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )

            left_vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            view = np.hstack((left_vis, depth_vis))

            torch.cuda.synchronize()
            fps = 1.0 / (time.perf_counter() - t0)
            cv2.putText(
                view,
                f"FFS {fps:.1f} FPS",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D456 IR (left) | FFS depth", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
