# SPDX-License-Identifier: Apache-2.0
"""A minimal binding to the CUDA runtime, for TensorRT device buffers and streams.

TensorRT's Python API takes raw device addresses and stream handles. Only these
few runtime calls are needed, so they are bound directly with ctypes; this keeps
TensorRT work independent of whether the installed PyTorch can use the GPU.

Buffers and streams are released by ``free()`` and ``destroy()``, by leaving a
``with`` block, or, as a last resort, when they are garbage collected.
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
    """A device allocation sized for one numpy array shape and dtype."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.address = 0  # Set first, so the safety net never sees a half-built buffer.
        self.shape = tuple(int(dimension) for dimension in shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape)) * self.dtype.itemsize
        pointer = ctypes.c_void_p()
        _check(_runtime().cudaMalloc(ctypes.byref(pointer), self.size), "cudaMalloc")
        self.address = int(pointer.value)

    def _require_allocated(self) -> None:
        if not self.address:
            raise RuntimeError("device buffer used after free()")

    def write(self, array: np.ndarray) -> None:
        """Copy a host array into this buffer, blocking until complete.

        The array must have this buffer's shape and a dtype of the same kind. It is
        made C-contiguous in this buffer's dtype first, because the copy is a flat
        run of bytes and a strided view would otherwise arrive in the wrong order.
        """
        self._require_allocated()
        array = np.asarray(array)
        if array.shape != self.shape:
            raise ValueError(f"expected an array of shape {self.shape}, got shape {array.shape}")
        if array.dtype.kind != self.dtype.kind:
            raise ValueError(f"expected an array of dtype {self.dtype}, got dtype {array.dtype}")
        contiguous = np.ascontiguousarray(array, dtype=self.dtype)
        if contiguous.nbytes != self.size:
            raise ValueError(
                f"expected {self.size} bytes for shape {self.shape} {self.dtype}, got {contiguous.nbytes}"
            )
        source = contiguous.ctypes.data_as(ctypes.c_void_p)
        _check(_runtime().cudaMemcpy(self.address, source, self.size, MEMCPY_HOST_TO_DEVICE), "cudaMemcpy")

    def read(self) -> np.ndarray:
        """Copy this buffer into a new host array, blocking until complete."""
        self._require_allocated()
        array = np.empty(self.shape, dtype=self.dtype)
        target = array.ctypes.data_as(ctypes.c_void_p)
        _check(_runtime().cudaMemcpy(target, self.address, self.size, MEMCPY_DEVICE_TO_HOST), "cudaMemcpy")
        return array

    def free(self) -> None:
        """Release the allocation; calling it again does nothing."""
        address, self.address = self.address, 0
        if address:
            _check(_runtime().cudaFree(address), "cudaFree")

    def __enter__(self) -> DeviceBuffer:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:  # A finaliser must not raise, and may run at interpreter shutdown.
            pass


class Stream:
    """A non-default CUDA stream, which TensorRT recommends over the default one."""

    def __init__(self) -> None:
        self.handle = 0  # Set first, so the safety net never sees a half-built stream.
        handle = ctypes.c_void_p()
        _check(_runtime().cudaStreamCreate(ctypes.byref(handle)), "cudaStreamCreate")
        self.handle = int(handle.value)

    def synchronize(self) -> None:
        if not self.handle:
            raise RuntimeError("stream used after destroy()")
        _check(_runtime().cudaStreamSynchronize(self.handle), "cudaStreamSynchronize")

    def destroy(self) -> None:
        """Destroy the stream; calling it again does nothing."""
        handle, self.handle = self.handle, 0
        if handle:
            _check(_runtime().cudaStreamDestroy(handle), "cudaStreamDestroy")

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:  # A finaliser must not raise, and may run at interpreter shutdown.
            pass
