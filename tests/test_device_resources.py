# SPDX-License-Identifier: Apache-2.0
"""Device buffers, streams and backends on the CPU, with the CUDA runtime and TensorRT faked.

The fake runtime keeps "device memory" in host byte arrays, so a copy in and a copy
out show exactly which bytes the real cudaMemcpy would have been given.
"""
import ctypes
import sys
import types
import warnings

import numpy as np
import pytest
import torch

from helpers import lively_detector
from peddet_orin import backends, cuda_native
from peddet_orin.export import INPUT_NAME, OUTPUT_NAMES

SHAPE = (2, 3)
SEED = 77
# Shaped like PyTorch's warning on its first CUDA use on this board.
CAPABILITY_WARNING = "Found GPU0 Orin which is of compute capability (CC) 8.7."


class FakeRuntime:
    """Enough of libcudart for cuda_native: malloc, free, memcpy and streams."""

    def __init__(self) -> None:
        self.memory: dict[int, ctypes.Array] = {}
        self.freed: list[int] = []
        self.streams: set[int] = set()
        self.destroyed: list[int] = []
        self.next_handle = 0x1000

    def _handle(self) -> int:
        self.next_handle += 0x1000
        return self.next_handle

    def cudaMalloc(self, reference, size):
        address = self._handle()
        self.memory[address] = ctypes.create_string_buffer(size)
        reference._obj.value = address
        return 0

    def cudaFree(self, address):
        del self.memory[address]
        self.freed.append(address)
        return 0

    def cudaMemcpy(self, destination, source, size, kind):
        if kind == cuda_native.MEMCPY_HOST_TO_DEVICE:
            ctypes.memmove(self.memory[destination], source, size)
        else:
            ctypes.memmove(destination, self.memory[source], size)
        return 0

    def cudaStreamCreate(self, reference):
        handle = self._handle()
        self.streams.add(handle)
        reference._obj.value = handle
        return 0

    def cudaStreamSynchronize(self, handle):
        return 0

    def cudaStreamDestroy(self, handle):
        self.streams.remove(handle)
        self.destroyed.append(handle)
        return 0

    def cudaGetErrorString(self, status):
        return b"fake error"


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(cuda_native, "_runtime", lambda: fake)
    return fake


def device_bytes(runtime: FakeRuntime, buffer: cuda_native.DeviceBuffer) -> np.ndarray:
    raw = bytes(runtime.memory[buffer.address])
    return np.frombuffer(raw, dtype=buffer.dtype).reshape(buffer.shape)


def test_write_then_read_round_trips_a_contiguous_array(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    values = np.arange(6, dtype=np.float32).reshape(SHAPE)
    buffer.write(values)
    assert np.array_equal(buffer.read(), values)


def test_write_converts_a_transposed_view_to_logical_order(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    view = np.arange(6, dtype=np.float32).reshape(3, 2).T  # Shape (2, 3), not C-contiguous.
    assert not view.flags.c_contiguous
    buffer.write(view)
    assert np.array_equal(device_bytes(runtime, buffer), view)


def test_write_passes_a_c_contiguous_source_to_memcpy(runtime, monkeypatch):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    seen = []
    original = runtime.cudaMemcpy

    def spy(destination, source, size, kind):
        seen.append(source.value)
        return original(destination, source, size, kind)

    monkeypatch.setattr(runtime, "cudaMemcpy", spy)
    view = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]  # Strided (2, 3) slice.
    assert not view.flags.c_contiguous
    buffer.write(view)
    assert seen and seen[0] != view.ctypes.data  # The view's own memory was not handed over.
    assert np.array_equal(device_bytes(runtime, buffer), view)


def test_write_casts_a_float64_array_to_the_buffer_dtype(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    values = np.linspace(0.0, 1.0, 6).reshape(SHAPE)
    buffer.write(values)
    assert np.allclose(device_bytes(runtime, buffer), values)


def test_write_rejects_a_wrong_shape_with_the_same_byte_count(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    with pytest.raises(ValueError, match=r"expected an array of shape \(2, 3\), got shape \(3, 2\)"):
        buffer.write(np.zeros((3, 2), dtype=np.float32))


def test_write_rejects_another_dtype_kind_naming_both(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    with pytest.raises(ValueError, match="expected an array of dtype float32, got dtype int32"):
        buffer.write(np.zeros(SHAPE, dtype=np.int32))


def test_free_is_idempotent_and_blocks_later_use(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    address = buffer.address
    buffer.free()
    buffer.free()
    assert runtime.freed == [address]
    with pytest.raises(RuntimeError, match="after free"):
        buffer.read()


def test_buffer_and_stream_are_released_by_with_blocks(runtime):
    with cuda_native.DeviceBuffer(SHAPE, np.float32) as buffer, cuda_native.Stream() as stream:
        address, handle = buffer.address, stream.handle
    assert runtime.freed == [address]
    assert runtime.destroyed == [handle]


def test_garbage_collection_frees_a_forgotten_buffer_and_stream(runtime):
    buffer = cuda_native.DeviceBuffer(SHAPE, np.float32)
    stream = cuda_native.Stream()
    address, handle = buffer.address, stream.handle
    del buffer, stream
    assert runtime.freed == [address]
    assert runtime.destroyed == [handle]


# A fake TensorRT, just wide enough for TensorRtBackend.
ENGINE_SHAPES = {INPUT_NAME: (1, 3, 4, 4), OUTPUT_NAMES[0]: (1, 5, 4), OUTPUT_NAMES[1]: (1, 5, 3)}


class FakeContext:
    def __init__(self) -> None:
        self.addresses = {}

    def set_tensor_address(self, name, address):
        self.addresses[name] = address

    def execute_async_v3(self, handle):
        return True


class FakeEngine:
    def __init__(self, dtypes=None) -> None:
        self.dtypes = dtypes or {}

    def get_tensor_shape(self, name):
        return ENGINE_SHAPES[name]

    def get_tensor_dtype(self, name):
        return self.dtypes.get(name, "FLOAT")

    def create_execution_context(self):
        return FakeContext()


def fake_tensorrt(engine) -> types.ModuleType:
    module = types.ModuleType("tensorrt")
    module.loggers = []

    class Logger:
        def __init__(self) -> None:
            module.loggers.append(self)

    class Runtime:
        def __init__(self, logger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, data):
            return engine

    module.Logger = Logger
    module.Runtime = Runtime
    module.DataType = types.SimpleNamespace(FLOAT="FLOAT", HALF="HALF")
    return module


@pytest.fixture
def engine_file(tmp_path):
    path = tmp_path / "broken.engine"
    path.write_bytes(b"not an engine")
    return path


def install_tensorrt(monkeypatch, engine) -> types.ModuleType:
    module = fake_tensorrt(engine)
    monkeypatch.setitem(sys.modules, "tensorrt", module)
    monkeypatch.setattr(backends, "_tensorrt_logger", None)
    return module


def test_an_engine_that_does_not_deserialise_names_its_file(monkeypatch, runtime, engine_file):
    install_tensorrt(monkeypatch, None)
    with pytest.raises(RuntimeError, match="could not deserialise the engine .*broken.engine"):
        backends.TensorRtBackend(engine_file)
    assert runtime.memory == {} and runtime.streams == set()


def test_backends_share_one_tensorrt_logger(monkeypatch, runtime, engine_file):
    module = install_tensorrt(monkeypatch, FakeEngine())
    first = backends.TensorRtBackend(engine_file)
    second = backends.TensorRtBackend(engine_file)
    assert len(module.loggers) == 1
    assert first.runtime.logger is second.runtime.logger
    first.close()
    second.close()


def test_tensorrt_backend_close_frees_every_buffer_and_the_stream(monkeypatch, runtime, engine_file):
    install_tensorrt(monkeypatch, FakeEngine())
    with backends.TensorRtBackend(engine_file) as backend:
        outputs = backend(torch.zeros(ENGINE_SHAPES[INPUT_NAME]))
        assert [tuple(output.shape) for output in outputs] == [ENGINE_SHAPES[name] for name in OUTPUT_NAMES]
        assert len(runtime.memory) == 3 and len(runtime.streams) == 1
    assert runtime.memory == {} and runtime.streams == set()
    backend.close()  # A second close does nothing.
    with pytest.raises(RuntimeError, match="after close"):
        backend(torch.zeros(ENGINE_SHAPES[INPUT_NAME]))


@pytest.mark.parametrize("name", [INPUT_NAME, *OUTPUT_NAMES])
def test_an_engine_whose_io_is_not_float_is_refused_before_any_buffer(monkeypatch, runtime, engine_file, name):
    install_tensorrt(monkeypatch, FakeEngine({name: "HALF"}))
    with pytest.raises(RuntimeError, match=f"tensor {name} is HALF, not FLOAT"):
        backends.TensorRtBackend(engine_file)
    assert runtime.memory == {} and runtime.streams == set()


def test_pytorch_backend_turns_tf32_off(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    backends.PytorchBackend(lively_detector(SEED), "cpu")
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_moving_the_network_leaves_the_capability_warning_visible(monkeypatch):
    def warn_like_torch(self, device):
        warnings.warn(CAPABILITY_WARNING, UserWarning, stacklevel=2)
        return self

    monkeypatch.setattr(backends.SsdNetwork, "to", warn_like_torch)
    with pytest.warns(UserWarning, match=r"compute capability \(CC\) 8\.7"):
        backends.PytorchBackend(lively_detector(SEED), "cpu")
