"""Two-hand skeleton extraction with MediaPipe Hands.

Section IV-A extracts skeletons with MediaPipe Hands [6], which the paper picks
because it is light enough to run on the Jetson edge device. The extractor
writes one ``(T, 42, 3)`` array per video, with the left hand in joints ``0..20``
and the right hand in joints ``21..41``.

Frames where a hand is not detected are filled with zeros. That missing data is
exactly what InterHand Temporal Fusion reconstructs, so it must be preserved
rather than interpolated away here.

``mediapipe`` and ``opencv-python`` are optional dependencies; install them with
``pip install -e ".[extraction]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..graph.hand_graph import NUM_JOINTS, NUM_JOINTS_PER_HAND

PathLike = Union[str, Path]


class MediaPipeHandSkeletonExtractor:
    """Extract two-hand 3D skeleton sequences from video files.

    Args:
        use_world_landmarks: Use MediaPipe's metric world landmarks (metres,
            origin at the hand's approximate geometric centre) instead of
            normalised image landmarks. Metric coordinates are what make the
            distance matrix ``D`` of Eq. (2) meaningful, so this is the default.
        min_detection_confidence: Forwarded to MediaPipe Hands.
        min_tracking_confidence: Forwarded to MediaPipe Hands.
        model_complexity: Forwarded to MediaPipe Hands.
    """

    def __init__(
        self,
        use_world_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "mediapipe is required for skeleton extraction; "
                'install it with `pip install -e ".[extraction]"`'
            ) from error

        self.use_world_landmarks = use_world_landmarks
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "MediaPipeHandSkeletonExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return the ``(42, 3)`` skeleton of a single RGB frame."""
        skeleton = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        results = self._hands.process(frame_rgb)
        landmark_sets = (
            results.multi_hand_world_landmarks
            if self.use_world_landmarks
            else results.multi_hand_landmarks
        )
        if not landmark_sets or not results.multi_handedness:
            return skeleton

        for landmarks, handedness in zip(landmark_sets, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            offset = 0 if label == "Left" else NUM_JOINTS_PER_HAND
            for joint, landmark in enumerate(landmarks.landmark):
                skeleton[offset + joint] = (landmark.x, landmark.y, landmark.z)
        return skeleton

    def extract_video(self, video_path: PathLike) -> np.ndarray:
        """Return the ``(T, 42, 3)`` skeleton sequence of a video."""
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "opencv-python is required to read videos; "
                'install it with `pip install -e ".[extraction]"`'
            ) from error

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(self.process_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"no frames decoded from {video_path}")
        return np.stack(frames).astype(np.float32)


def sliding_windows(
    sequence: np.ndarray,
    window_size: int = 30,
    stride: Optional[int] = None,
    drop_last: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a ``(T, 42, 3)`` sequence into windows.

    Args:
        sequence: Skeleton sequence.
        window_size: Frames per window (30 in the paper, i.e. one second at 30 fps).
        stride: Hop between windows. Defaults to ``window_size`` (no overlap),
            which keeps the windows independent for the temporal-order split.
        drop_last: Discard a trailing partial window instead of padding it.

    Returns:
        ``(windows, start_frames)`` with shapes ``(S, window_size, 42, 3)`` and ``(S,)``.
    """
    stride = window_size if stride is None else stride
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    starts = list(range(0, max(len(sequence) - window_size + 1, 0), stride))
    if not starts and not drop_last and len(sequence) > 0:
        padded = np.zeros((window_size, *sequence.shape[1:]), dtype=np.float32)
        padded[: len(sequence)] = sequence
        return padded[None], np.zeros(1, dtype=np.int64)

    windows = np.stack([sequence[start : start + window_size] for start in starts])
    return windows.astype(np.float32), np.asarray(starts, dtype=np.int64)
