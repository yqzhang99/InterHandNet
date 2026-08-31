#!/usr/bin/env python3
"""Train InterHandNet on a single cross-validation fold.

Example:
    python tools/train.py --config configs/interhandnet.yaml --camera 100 --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.apis import build_dataset, resolve_device, train_fold  # noqa: E402
from interhandnet.data.splits import group_indices_by_camera, temporal_kfold  # noqa: E402
from interhandnet.utils import apply_overrides, load_config, save_config, set_seed  # noqa: E402
from interhandnet.utils.logger import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--camera", default=None, help="camera setting to train on")
    parser.add_argument("--fold", type=int, default=0, help="validation fold index")
    parser.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu"')
    parser.add_argument("--work-dir", default=None, help="overrides work_dir from the config")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. training.epochs=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    work_dir = Path(args.work_dir or config.get("work_dir", "work_dirs/run"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=work_dir / "train.log")
    set_seed(config.get("seed", 1), config.get("deterministic", False))
    save_config(config, work_dir / "config.yaml")

    dataset = build_dataset(config["data"])
    logger.info(f"loaded {len(dataset)} windows from {config['data']['archive']}")

    groups = group_indices_by_camera(
        dataset.cameras, dataset.start_frames, num_samples=len(dataset)
    )
    if args.camera is not None:
        matching = [key for key in groups if str(key) == str(args.camera)]
        if not matching:
            raise SystemExit(
                f"camera {args.camera!r} not in archive; available: {sorted(map(str, groups))}"
            )
        indices = groups[matching[0]]
    else:
        indices = next(iter(groups.values())) if len(groups) == 1 else None
        if indices is None:
            raise SystemExit(
                "the archive holds several camera settings; pass --camera or use "
                "tools/cross_validate.py"
            )

    cv_config = config.get("cross_validation", {})
    folds = temporal_kfold(
        indices, num_folds=cv_config.get("num_folds", 5), gap=cv_config.get("gap", 0)
    )
    if not 0 <= args.fold < len(folds):
        raise SystemExit(f"--fold must be in [0, {len(folds) - 1}], got {args.fold}")
    train_indices, validation_indices = folds[args.fold]
    logger.info(
        f"camera {args.camera} fold {args.fold}: "
        f"{len(train_indices)} train / {len(validation_indices)} validation windows"
    )

    trainer, metrics = train_fold(
        config,
        dataset,
        train_indices,
        validation_indices,
        device=resolve_device(args.device),
        logger=logger.info,
    )

    if metrics is not None:
        logger.info(f"best epoch {trainer.best_epoch}: {metrics.format_summary()}")
    trainer.save_checkpoint(
        work_dir / "best.pt",
        extra={"config": config, "camera": args.camera, "fold": args.fold},
    )


if __name__ == "__main__":
    main()
