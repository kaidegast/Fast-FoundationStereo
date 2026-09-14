#!/usr/bin/env python3
"""Build a static Fast-FoundationStereo TensorRT engine from a single ONNX.

This uses TensorRT's Python builder API so it also works in Jetson containers
that package the TensorRT bindings but not the ``trtexec`` sample binary.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default="output_ffs_trt_jetson/fast_foundationstereo.onnx",
        help="Input static ONNX model",
    )
    parser.add_argument(
        "--engine",
        default="output_ffs_trt_jetson/fast_foundationstereo.engine",
        help="Output TensorRT engine",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=2.0,
        help="TensorRT builder workspace in GiB",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Build FP32 instead of the default FP16 engine",
    )
    args = parser.parse_args()

    import tensorrt as trt

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    onnx_parser = trt.OnnxParser(network, logger)

    if not onnx_parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(onnx_parser.get_error(i))
            for i in range(onnx_parser.num_errors)
        )
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if not args.fp32:
        if not builder.platform_has_fast_fp16:
            print("Warning: platform does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    workspace_bytes = int(args.workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    precision = "FP32" if args.fp32 else "FP16"
    print(
        f"Building {precision} engine from {onnx_path} "
        f"with {args.workspace_gb:g} GiB workspace..."
    )
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"Saved {engine_path} ({engine_path.stat().st_size / (1 << 20):.1f} MiB)")


if __name__ == "__main__":
    main()
