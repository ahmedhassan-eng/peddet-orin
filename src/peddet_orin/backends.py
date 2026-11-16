# SPDX-License-Identifier: Apache-2.0
"""Inference backends: the same call for PyTorch FP32 and TensorRT engines.

Each backend takes a normalised CPU batch and returns CPU tensors, so evaluation
and benchmarks treat every precision identically. ``label`` names what ran.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from peddet_orin import cuda_native
from peddet_orin.export import INPUT_NAME, OUTPUT_NAMES, display_path
from peddet_orin.model import SsdNetwork, use_strict_fp32

# One TensorRT logger for the whole process: TensorRT warns when a second one is passed in.
_tensorrt_logger = None


def tensorrt_logger():
    """Return the process-wide TensorRT logger, creating it on first use."""
    global _tensorrt_logger
    if _tensorrt_logger is None:
        import tensorrt  # Imported here so CPU-only code never needs TensorRT.

        _tensorrt_logger = tensorrt.Logger()
    return _tensorrt_logger


class PytorchBackend:
    """Run the network in FP32 on a chosen torch device, with TF32 turned off."""

    def __init__(self, detector: torch.nn.Module, device: str) -> None:
        self.device = device
        self.label = f"pytorch-{device}"
        use_strict_fp32()
        self.network = SsdNetwork(detector).to(device).eval()

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            box_regression, class_logits = self.network(batch.to(self.device))
        return box_regression.cpu(), class_logits.cpu()


class TensorRtBackend:
    """Run a serialised TensorRT engine through explicit device buffers.

    Call ``close()``, or use the backend as a context manager, to free the device
    buffers and the stream; garbage collection frees them otherwise.
    """

    label = "tensorrt-gpu"

    def __init__(self, engine_path: Path) -> None:
        import tensorrt  # Imported here so CPU-only tests never need TensorRT.

        self.buffers: dict[str, cuda_native.DeviceBuffer] = {}
        self.stream: cuda_native.Stream | None = None
        self.context = None
        self.engine = None
        self.runtime = tensorrt.Runtime(tensorrt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT could not deserialise the engine {display_path(engine_path)}; "
                "rebuild it with this TensorRT version on this device"
            )
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda_native.Stream()
            for name in (INPUT_NAME, *OUTPUT_NAMES):
                # The buffers hold float32; an engine with any other I/O type would read or write them wrongly.
                dtype = self.engine.get_tensor_dtype(name)
                if dtype != tensorrt.DataType.FLOAT:
                    raise RuntimeError(
                        f"engine {display_path(engine_path)} tensor {name} is {dtype}, not FLOAT; "
                        "rebuild it with python -m peddet_orin.engine_build"
                    )
                shape = tuple(self.engine.get_tensor_shape(name))
                buffer = cuda_native.DeviceBuffer(shape, np.float32)
                self.buffers[name] = buffer
                self.context.set_tensor_address(name, buffer.address)
        except BaseException:
            self.close()
            raise

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stream is None:
            raise RuntimeError("TensorRT backend used after close()")
        self.buffers[INPUT_NAME].write(batch.numpy())
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT rejected the inference")
        self.stream.synchronize()
        box_regression, class_logits = (torch.from_numpy(self.buffers[name].read()) for name in OUTPUT_NAMES)
        return box_regression, class_logits

    def close(self) -> None:
        """Free the device buffers and destroy the stream; calling it again does nothing."""
        buffers, self.buffers = self.buffers, {}
        for buffer in buffers.values():
            buffer.free()
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.destroy()
        self.context = None
        self.engine = None

    def __enter__(self) -> TensorRtBackend:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # A finaliser must not raise, and may run at interpreter shutdown.
            pass
