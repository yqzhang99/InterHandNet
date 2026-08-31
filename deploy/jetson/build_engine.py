#!/usr/bin/env python3
"""Convert an ONNX model into a TensorRT engine.

Section IV-C deploys InterHandNet on an NVIDIA Jetson Orin Nano through
TensorRT. Run this script on the Jetson itself: an engine is tied to the GPU
architecture and the TensorRT version it was built with.

Example:
    python deploy/jetson/build_engine.py --onnx interhandnet.onnx \\
        --engine interhandnet.engine --precision fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="input ONNX model")
    parser.add_argument("--engine", required=True, help="output engine file")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp32", "fp16"),
        help="fp16 is the usual choice on Jetson",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="build an optimisation profile for a dynamic batch dimension",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 30,
        help="scratch memory in bytes available to the builder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorrt as trt
    except ImportError as error:  # pragma: no cover - Jetson-only dependency
        raise SystemExit(
            "TensorRT is not available. It ships with the JetPack SDK on Jetson "
            "devices and cannot be installed from PyPI."
        ) from error

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX model not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("warning: this platform has no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    if args.max_batch_size is not None:
        tensor = network.get_input(0)
        shape = list(tensor.shape)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            tensor.name,
            min=(1, *shape[1:]),
            opt=(1, *shape[1:]),
            max=(args.max_batch_size, *shape[1:]),
        )
        config.add_optimization_profile(profile)

    print(f"building engine from {onnx_path} ({args.precision}); this can take a few minutes")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")

    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"engine written to {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
