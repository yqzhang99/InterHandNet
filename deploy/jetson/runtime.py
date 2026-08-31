"""Minimal TensorRT runtime wrapper for InterHandNet.

Keeps a single engine, one execution context and pre-allocated device buffers, so
that a real-time loop only pays for the memory copies and the inference itself.

The module lives in ``deploy/jetson`` rather than ``deploy/tensorrt`` on purpose:
a package directory named ``tensorrt`` would shadow the real TensorRT package for
any process started from ``deploy/``.

TensorRT and pycuda ship with the JetPack SDK; they are not installable from
PyPI and are imported lazily so that the rest of the repository stays usable on
a workstation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


class TensorRTPredictor:
    """Run a serialized TensorRT engine on skeleton windows.

    Args:
        engine_path: Engine produced by ``deploy/tensorrt/build_engine.py``.

    Shape:
        - Input: ``(N, 3, T, 42)`` float32, matching the exported ONNX model.
        - Output: ``(N, num_classes)`` float32 logits.
    """

    def __init__(self, engine_path: PathLike) -> None:
        try:
            import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as error:  # pragma: no cover - Jetson-only dependency
            raise ImportError(
                "TensorRT and pycuda are required; they are part of the JetPack SDK"
            ) from error

        self._cuda = cuda
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"engine not found: {engine_path}")

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self._context = self._engine.create_execution_context()

        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        self.input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        self.output_shape = tuple(self._engine.get_tensor_shape(self._output_name))

        self._host_output = cuda.pagelocked_empty(
            int(np.prod(self.output_shape)), dtype=np.float32
        )
        self._device_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)
        self._device_output = cuda.mem_alloc(self._host_output.nbytes)
        self._stream = cuda.Stream()

    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        return self.predict(skeleton)

    def predict(self, skeleton: np.ndarray) -> np.ndarray:
        """Return the logits for one batch of skeleton windows."""
        skeleton = np.ascontiguousarray(skeleton, dtype=np.float32)
        if skeleton.shape != self.input_shape:
            raise ValueError(
                f"engine expects input shape {self.input_shape}, got {skeleton.shape}"
            )

        self._cuda.memcpy_htod_async(self._device_input, skeleton, self._stream)
        self._context.set_tensor_address(self._input_name, int(self._device_input))
        self._context.set_tensor_address(self._output_name, int(self._device_output))
        self._context.execute_async_v3(stream_handle=self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._host_output, self._device_output, self._stream)
        self._stream.synchronize()
        return self._host_output.reshape(self.output_shape).copy()
