"""A minimal binding to the CUDA runtime, for TensorRT device buffers and streams.

TensorRT's Python API takes raw device addresses and stream handles. Only these
few runtime calls are needed, so they are bound directly with ctypes; this keeps
TensorRT work independent of whether the installed PyTorch can use the GPU.
"""
from __future__ import annotations

import ctypes

import numpy as np

# cudaMalloc and friends live in the CUDA runtime, not in the driver library.
CUDA_RUNTIME_LIBRARIES = ("libcudart.so.13", "libcudart.so")
CUDA_SUCCESS = 0
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2

_library: ctypes.CDLL | None = None


def _open_first(names: tuple[str, ...]) -> ctypes.CDLL:
    """Open the first library that loads, reporting every name on total failure."""
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(f"no CUDA runtime library found among {names}")


def _runtime() -> ctypes.CDLL:
    """Load the runtime on first use, so importing this module needs no GPU."""
    global _library
    if _library is None:
        library = _open_first(CUDA_RUNTIME_LIBRARIES)
        library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        library.cudaFree.argtypes = [ctypes.c_void_p]
        library.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        library.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        library.cudaGetErrorString.restype = ctypes.c_char_p
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        _library = library
    return _library


def _check(status: int, call: str) -> None:
    if status != CUDA_SUCCESS:
        reason = _runtime().cudaGetErrorString(status).decode()
        raise RuntimeError(f"{call} failed: {reason}")


class DeviceBuffer:
    """A device allocation sized for one numpy array shape."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(shape)) * self.dtype.itemsize
        pointer = ctypes.c_void_p()
        _check(_runtime().cudaMalloc(ctypes.byref(pointer), self.size), "cudaMalloc")
        self.address = int(pointer.value)

    def write(self, array: np.ndarray) -> None:
        """Copy a host array into this buffer, blocking until complete."""
        if array.nbytes != self.size:
            raise ValueError(f"expected {self.size} bytes, got {array.nbytes}")
        source = array.ctypes.data_as(ctypes.c_void_p)
        _check(_runtime().cudaMemcpy(self.address, source, self.size, MEMCPY_HOST_TO_DEVICE), "cudaMemcpy")

    def read(self) -> np.ndarray:
        """Copy this buffer into a new host array, blocking until complete."""
        array = np.empty(self.shape, dtype=self.dtype)
        target = array.ctypes.data_as(ctypes.c_void_p)
        _check(_runtime().cudaMemcpy(target, self.address, self.size, MEMCPY_DEVICE_TO_HOST), "cudaMemcpy")
        return array

    def free(self) -> None:
        _check(_runtime().cudaFree(self.address), "cudaFree")


class Stream:
    """A non-default CUDA stream, which TensorRT recommends over the default one."""

    def __init__(self) -> None:
        handle = ctypes.c_void_p()
        _check(_runtime().cudaStreamCreate(ctypes.byref(handle)), "cudaStreamCreate")
        self.handle = int(handle.value)

    def synchronize(self) -> None:
        _check(_runtime().cudaStreamSynchronize(self.handle), "cudaStreamSynchronize")

    def destroy(self) -> None:
        _check(_runtime().cudaStreamDestroy(self.handle), "cudaStreamDestroy")
