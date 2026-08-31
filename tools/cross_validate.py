#!/usr/bin/env python3
"""Run the full evaluation protocol of Section IV-B.

Every camera setting is cross-validated with five temporally ordered folds, and
the per-camera results are combined into a weighted average that balances the
different dataset sizes across settings.

Example:
    python tools/cross_validate.py --config configs/interhandnet.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.apis import build_dataset, resolve_device, train_fold  # noqa: E402
from interhandnet.data.splits import camera_wise_folds, weighted_average  # noqa: E402
from interhandnet.engine.metrics import METRIC_NAMES  # noqa: E402
from interhandnet.utils import apply_overrides, load_config, save_config, set_seed  # noqa: E402
from interhandnet.utils.logger import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu"')
    parser.add_argument("--work-dir", default=None, help="overrides work_dir from the config")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="keep the best checkpoint of every fold instead of metrics only",
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    work_dir = Path(args.work_dir or config.get("work_dir", "work_dirs/cross_validation"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=work_dir / "cross_validate.log")
    set_seed(config.get("seed", 1), config.get("deterministic", False))
    save_config(config, work_dir / "config.yaml")

    dataset = build_dataset(config["data"])
    device = resolve_device(args.device)
    cv_config = config.get("cross_validation", {})

    fold_records = []
    per_camera = defaultdict(list)

    for camera, fold, train_indices, validation_indices in camera_wise_folds(
        dataset.cameras,
        dataset.start_frames,
        num_folds=cv_config.get("num_folds", 5),
        gap=cv_config.get("gap", 0),
        num_samples=len(dataset),
    ):
        logger.info(
            f"=== camera {camera} fold {fold}: {len(train_indices)} train / "
            f"{len(validation_indices)} validation windows ==="
        )
        trainer, metrics = train_fold(
            config,
            dataset,
            train_indices,
            validation_indices,
            device=device,
            logger=logger.info,
        )
        if metrics is None:
            raise RuntimeError("validation produced no metrics")

        logger.info(f"camera {camera} fold {fold} best: {metrics.format_summary()}")
        record = {"camera": str(camera), "fold": fold, **metrics.to_dict()}
        fold_records.append(record)
        per_camera[str(camera)].append(metrics)

        if args.save_checkpoints:
            trainer.save_checkpoint(
                work_dir / f"camera_{camera}_fold_{fold}.pt",
                extra={"config": config, "camera": camera, "fold": fold},
            )

    camera_summary = {}
    for camera, metrics_list in per_camera.items():
        sample_counts = [m.num_samples for m in metrics_list]
        camera_summary[camera] = {
            name: weighted_average([getattr(m, name) for m in metrics_list], sample_counts)
            for name in METRIC_NAMES
        }
        camera_summary[camera]["num_samples"] = int(sum(sample_counts))
        logger.info(
            f"camera {camera} mean over folds: "
            + "  ".join(f"{name} {camera_summary[camera][name]:.4f}" for name in METRIC_NAMES)
        )

    camera_weights = [camera_summary[camera]["num_samples"] for camera in camera_summary]
    overall = {
        name: weighted_average(
            [camera_summary[camera][name] for camera in camera_summary], camera_weights
        )
        for name in METRIC_NAMES
    }
    logger.info(
        "weighted average over camera settings: "
        + "  ".join(f"{name} {overall[name]:.4f}" for name in METRIC_NAMES)
    )

    confusion = np.sum(
        [m.confusion_matrix for metrics_list in per_camera.values() for m in metrics_list], axis=0
    )
    results_path = work_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "folds": fold_records,
                "per_camera": camera_summary,
                "overall": overall,
                "confusion_matrix": confusion.tolist(),
            },
            handle,
            indent=2,
        )
    logger.info(f"results written to {results_path}")


if __name__ == "__main__":
    main()
