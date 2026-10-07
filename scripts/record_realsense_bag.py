#!/usr/bin/env python3
"""Record a synchronized RealSense IR stereo sequence for offline comparison."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record D456 infrared streams to a RealSense .bag file."
    )
    parser.add_argument("--output", required=True, help="Output .bag recording")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--show", action="store_true", help="show the left IR stream while recording"
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".bag":
        parser.error("--output must end in .bag")
    if output.exists():
        parser.error(f"refusing to overwrite existing recording: {output}")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps
    )
    config.enable_stream(
        rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps
    )
    config.enable_record_to_file(str(output.resolve()))

    started = time.monotonic()
    frames_recorded = 0
    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print("IR emitter enabled:", depth_sensor.get_option(rs.option.emitter_enabled))

        print(f"Recording {args.seconds:g} seconds to {output} ...")
        while time.monotonic() - started < args.seconds:
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            if not left_frame:
                continue
            frames_recorded += 1

            if args.show:
                left = np.asanyarray(left_frame.get_data())
                cv2.imshow("D456 left IR — recording", left)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("Recording interrupted; finalizing .bag file.")
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    print(
        f"Saved {output} ({frames_recorded} frames, {elapsed:.1f} s, "
        f"{frames_recorded / elapsed:.1f} FPS)."
    )


if __name__ == "__main__":
    main()
