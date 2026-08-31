#!/usr/bin/env python3
"""Export a trained model to ONNX for TensorRT deployment.

Section IV-C saves the parameters of the best epoch in ONNX format and then
converts them into a TensorRT engine for the Jetson. The default input shape is
``(1, 3, 30, 42)``: one second of two-hand skeleton data at 30 fps.

Example:
    python tools/export_onnx.py --config configs/interhandnet.yaml \\
        --checkpoint work_dirs/interhandnet/best.pt --output interhandnet.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.apis import load_checkpoint, resolve_device  # noqa: E402
from interhandnet.graph import NUM_JOINTS  # noqa: E402
from interhandnet.utils import apply_overrides, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="interhandnet.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="frames per window; defaults to data.window_size from the config",
    )
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="mark the batch dimension as dynamic",
    )
    parser.add_argument("--device", default="cpu", help="export device, cpu is usually enough")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)

    window_size = args.window_size or config["data"].get("window_size", 30)
    in_channels = config["model"].get("in_channels", 3)

    model = load_checkpoint(args.checkpoint, config, device)
    dummy_input = torch.zeros(
        args.batch_size, in_channels, window_size, NUM_JOINTS, device=device
    )

    dynamic_axes = {"skeleton": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["skeleton"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"exported {output_path} with input shape {tuple(dummy_input.shape)}")

    try:
        import onnx
    except ImportError:
        print("install onnx to validate the exported graph: pip install onnx")
        return

    onnx.checker.check_model(onnx.load(str(output_path)))
    print("ONNX graph check passed")


if __name__ == "__main__":
    main()
