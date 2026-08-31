#!/usr/bin/env python3
"""Build a skeleton archive from a directory of hand-washing videos.

The archive layout is documented in
:class:`interhandnet.data.dataset.HandWashingSkeletonDataset`.

Labels and camera settings are taken from the file paths. The default pattern
matches the layout of the dataset by Lulla et al., where each video lives in a
per-camera directory and its movement code is encoded in the file name:

    <root>/<camera>/<something>_<code>.mp4

The movement code is used as the class index directly: ``0`` is "other movement"
and ``1..6`` are the six WHO steps. Adjust ``--label-regex``,
``--camera-regex`` and ``--label-offset`` for a different naming scheme.

Example:
    python tools/extract_skeletons.py --videos data/raw --output data/handwashing_mediapipe.npz
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interhandnet.data.skeleton_extraction import (  # noqa: E402
    MediaPipeHandSkeletonExtractor,
    sliding_windows,
)

DEFAULT_LABEL_REGEX = r"_(\d+)\.[^.]+$"
DEFAULT_CAMERA_REGEX = r"(?:^|/)(\d{3})(?:/|$)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, help="root directory holding the videos")
    parser.add_argument("--output", required=True, help="destination .npz archive")
    parser.add_argument("--pattern", default="**/*.mp4", help="glob applied under --videos")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="hop between windows; defaults to --window-size (no overlap)",
    )
    parser.add_argument("--label-regex", default=DEFAULT_LABEL_REGEX)
    parser.add_argument("--camera-regex", default=DEFAULT_CAMERA_REGEX)
    parser.add_argument(
        "--label-offset",
        type=int,
        default=0,
        help=(
            "value subtracted from the parsed label; the default of 0 keeps the "
            "dataset's movement codes (0 = other, 1..6 = WHO steps) as class "
            "indices. Use 1 if your file names encode 1-based steps only."
        ),
    )
    parser.add_argument(
        "--image-landmarks",
        action="store_true",
        help="store normalised image landmarks instead of metric world landmarks",
    )
    return parser.parse_args()


def search(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    root = Path(args.videos)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    videos = sorted(root.glob(args.pattern))
    if not videos:
        raise SystemExit(f"no videos matched {args.pattern!r} under {root}")
    print(f"found {len(videos)} videos")

    windows: List[np.ndarray] = []
    labels: List[int] = []
    cameras: List[str] = []
    session_ids: List[str] = []
    start_frames: List[int] = []
    skipped = 0
    # Frame positions are accumulated across videos so that sorting by
    # `start_frames` reproduces the global recording order, which is what the
    # temporal-order split of Section IV-B needs.
    global_offset = 0

    with MediaPipeHandSkeletonExtractor(
        use_world_landmarks=not args.image_landmarks
    ) as extractor:
        for index, video in enumerate(videos, start=1):
            relative = video.relative_to(root).as_posix()
            raw_label = search(args.label_regex, relative)
            if raw_label is None:
                print(f"[{index}/{len(videos)}] no label in {relative}, skipping")
                skipped += 1
                continue

            label = int(raw_label) - args.label_offset
            if label < 0:
                print(f"[{index}/{len(videos)}] negative label in {relative}, skipping")
                skipped += 1
                continue

            camera = search(args.camera_regex, relative) or "unknown"
            sequence = extractor.extract_video(video)
            video_windows, starts = sliding_windows(
                sequence, window_size=args.window_size, stride=args.stride
            )
            if len(video_windows) == 0:
                print(f"[{index}/{len(videos)}] too short: {relative}, skipping")
                skipped += 1
                continue

            windows.append(video_windows)
            labels.extend([label] * len(video_windows))
            cameras.extend([camera] * len(video_windows))
            session_ids.extend([relative] * len(video_windows))
            start_frames.extend((starts + global_offset).tolist())
            global_offset += len(sequence)
            print(
                f"[{index}/{len(videos)}] {relative}: {len(sequence)} frames -> "
                f"{len(video_windows)} windows (class {label}, camera {camera})"
            )

    if not windows:
        raise SystemExit("no windows extracted")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        skeletons=np.concatenate(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        cameras=np.asarray(cameras),
        session_ids=np.asarray(session_ids),
        start_frames=np.asarray(start_frames, dtype=np.int64),
    )
    print(f"wrote {len(labels)} windows to {output} ({skipped} videos skipped)")


if __name__ == "__main__":
    main()
