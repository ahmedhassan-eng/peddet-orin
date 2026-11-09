"""Inference backends: the same call for PyTorch FP32 and TensorRT engines.

Each backend takes a normalised CPU batch and returns CPU tensors, so evaluation
and benchmarks treat every precision identically. ``label`` names what ran.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from peddet_orin import cuda_native
from peddet_orin.export import INPUT_NAME, OUTPUT_NAMES
from peddet_orin.model import SsdNetwork, to_device


class PytorchBackend:
    """Run the network in FP32 on a chosen torch device."""

    def __init__(self, detector: torch.nn.Module, device: str) -> None:
        self.device = device
        self.label = f"pytorch-{device}"
        self.network = to_device(SsdNetwork(detector), device).eval()

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            box_regression, class_logits = self.network(batch.to(self.device))
        return box_regression.cpu(), class_logits.cpu()


class TensorRtBackend:
    """Run a serialised TensorRT engine through explicit device buffers."""

    label = "tensorrt-gpu"

    def __init__(self, engine_path: Path) -> None:
        import tensorrt  # Imported here so CPU-only tests never need TensorRT.

        self.runtime = tensorrt.Runtime(tensorrt.Logger())
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = cuda_native.Stream()
        self.buffers = {}
        for name in (INPUT_NAME, *OUTPUT_NAMES):
            shape = tuple(self.engine.get_tensor_shape(name))
            buffer = cuda_native.DeviceBuffer(shape, np.float32)
            self.context.set_tensor_address(name, buffer.address)
            self.buffers[name] = buffer

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.buffers[INPUT_NAME].write(batch.numpy())
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT rejected the inference")
        self.stream.synchronize()
        box_regression, class_logits = (torch.from_numpy(self.buffers[name].read()) for name in OUTPUT_NAMES)
        return box_regression, class_logits
