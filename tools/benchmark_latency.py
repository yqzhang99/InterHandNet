#!/usr/bin/env python3
"""Measure the computational latency reported in Table VI.

The paper's protocol: process 1,800 samples of shape ``(1, 3, 30, 42)``, start
the clock after the model is loaded and before the first prediction, stop it
after the last prediction, and divide by the number of samples. The first five
samples are excluded to remove the warm-up phase.

Example:
    python tools/benchmark_latency.py --config configs/interhandnet.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.apis import resolve_device  # noqa: E402
from interhandnet.graph import NUM_JOINTS  # noqa: E402
from interhandnet.models import build_model  # noqa: E402
from interhandnet.utils import apply_overrides, load_config  # noqa: E402

WARMUP_SAMPLES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="optional trained weights")
    parser.add_argument("--num-samples", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)

    window_size = args.window_size or config["data"].get("window_size", 30)
    in_channels = config["model"].get("in_channels", 3)

    model = build_model(config["model"]).to(device).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload.get("state_dict", payload))

    sample = torch.randn(args.batch_size, in_channels, window_size, NUM_JOINTS, device=device)
    is_cuda = device.type == "cuda"

    for _ in range(WARMUP_SAMPLES):
        model(sample)
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.num_samples):
        model(sample)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_sample_ms = elapsed / args.num_samples * 1000.0
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device                {device}")
    print(f"input shape           {tuple(sample.shape)}")
    print(f"parameters            {parameters / 1e6:.3f} M")
    print(f"samples               {args.num_samples} (after {WARMUP_SAMPLES} warm-up runs)")
    print(f"total time            {elapsed:.3f} s")
    print(f"latency per sample    {per_sample_ms:.3f} ms")


if __name__ == "__main__":
    main()
