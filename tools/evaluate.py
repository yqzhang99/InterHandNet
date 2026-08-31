#!/usr/bin/env python3
"""Evaluate a checkpoint and print the confusion matrix.

Example:
    python tools/evaluate.py --config configs/interhandnet.yaml \\
        --checkpoint work_dirs/interhandnet/best.pt --camera 100 --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.apis import build_dataset, build_loader, load_checkpoint, resolve_device  # noqa: E402
from interhandnet.data.dataset import class_names  # noqa: E402
from interhandnet.data.splits import group_indices_by_camera, temporal_kfold  # noqa: E402
from interhandnet.engine import evaluate  # noqa: E402
from interhandnet.utils import apply_overrides, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", default=None, help="restrict to one camera setting")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="evaluate the validation split of this fold; omit to use all samples",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def format_confusion_matrix(matrix: np.ndarray) -> str:
    header = "        " + "".join(f"{index + 1:>8d}" for index in range(matrix.shape[1]))
    rows = [header]
    for index, row in enumerate(matrix):
        rows.append(f"step {index + 1:d} " + "".join(f"{value:>8d}" for value in row))
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)

    dataset = build_dataset(config["data"])
    indices = np.arange(len(dataset))

    if args.camera is not None:
        groups = group_indices_by_camera(
            dataset.cameras, dataset.start_frames, num_samples=len(dataset)
        )
        matching = [key for key in groups if str(key) == str(args.camera)]
        if not matching:
            raise SystemExit(
                f"camera {args.camera!r} not in archive; available: {sorted(map(str, groups))}"
            )
        indices = groups[matching[0]]

    if args.fold is not None:
        cv_config = config.get("cross_validation", {})
        folds = temporal_kfold(
            indices, num_folds=cv_config.get("num_folds", 5), gap=cv_config.get("gap", 0)
        )
        if not 0 <= args.fold < len(folds):
            raise SystemExit(f"--fold must be in [0, {len(folds) - 1}], got {args.fold}")
        indices = folds[args.fold][1]

    loader = build_loader(
        dataset.subset(indices),
        batch_size=config["data"].get("batch_size", 64),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
    )
    model = load_checkpoint(args.checkpoint, config, device)
    metrics = evaluate(model, loader, config["model"]["num_classes"], device=device)

    print(metrics.format_summary())
    print("\nper-class F1")
    names = class_names(config["model"]["num_classes"])
    for index, score in enumerate(metrics.per_class_f1):
        print(f"  {names[index]:<48s} {score:.4f}")
    print("\nconfusion matrix (rows: ground truth, columns: prediction)")
    print(format_confusion_matrix(metrics.confusion_matrix))


if __name__ == "__main__":
    main()
