#!/usr/bin/env python3
"""Real-time hand-washing step recognition on an edge device.

Implements the workflow of Section IV-C: capture RGB-D video with a depth
camera, extract 3D hand skeletons with MediaPipe Hands, and run the
TensorRT-optimised InterHandNet once per second.

Example:
    python deploy/jetson/realtime_demo.py --engine interhandnet.engine --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.jetson.runtime import TensorRTPredictor  # noqa: E402
from interhandnet.data.dataset import class_names  # noqa: E402
from interhandnet.data.skeleton_extraction import MediaPipeHandSkeletonExtractor  # noqa: E402
from interhandnet.data.transforms import to_model_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="TensorRT engine file")
    parser.add_argument(
        "--camera", default="0", help="OpenCV camera index or a video file to replay"
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--display", action="store_true", help="show the camera feed with the prediction overlaid"
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("opencv-python is required for the camera loop") from error

    source = int(args.camera) if args.camera.isdigit() else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera source: {args.camera}")

    predictor = TensorRTPredictor(args.engine)
    print(f"engine input shape {predictor.input_shape}")

    frames: deque = deque(maxlen=args.window_size)
    last_label = "collecting frames"
    last_confidence = 0.0

    with MediaPipeHandSkeletonExtractor(use_world_landmarks=True) as extractor:
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frames.append(
                    extractor.process_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                )

                if len(frames) == args.window_size:
                    window = to_model_layout(np.stack(frames))[None]
                    start = time.perf_counter()
                    probabilities = softmax(predictor.predict(window)[0])
                    latency_ms = (time.perf_counter() - start) * 1000.0

                    step = int(probabilities.argmax())
                    last_confidence = float(probabilities[step])
                    last_label = class_names(len(probabilities))[step]
                    print(f"{last_label}  p={last_confidence:.2f}  ({latency_ms:.1f} ms)")
                    frames.clear()

                if args.display:
                    cv2.putText(
                        frame_bgr,
                        f"{last_label} ({last_confidence:.2f})",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("InterHandNet", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.release()
            if args.display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
