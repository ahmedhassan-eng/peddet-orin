"""The GPU check's slice inference, CLI and log fields, on the CPU with CUDA calls faked."""
import warnings

import pytest
import torch

from peddet_orin import gpu_check

ARCH_LIST = ["sm_80", "sm_87", "sm_90", "sm_100", "sm_110", "sm_120"]
# Shaped like the warning PyTorch emits on this board; the test only needs it to be passed through unchanged.
TORCH_WARNING = "Found GPU0 Orin which is of compute capability (CC) 8.7. The following list shows the CCs this version"


@pytest.mark.parametrize(
    ("arch", "capability"),
    [("sm_80", (8, 0)), ("sm_87", (8, 7)), ("sm_90", (9, 0)), ("sm_100", (10, 0)), ("sm_110", (11, 0)),
     ("sm_120", (12, 0)), ("compute_87", None)],
)
def test_slice_names_parse_as_one_integer_capability(arch, capability):
    assert gpu_check.slice_capability(arch) == capability


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [(8, 7, "sm_87"), (10, 0, "sm_100"), (11, 0, "sm_110"), (12, 0, "sm_120"), (1, 0, "none")],
)
def test_compatible_slice_handles_two_digit_majors(major, minor, expected):
    assert gpu_check.compatible_slice(ARCH_LIST, major, minor) == expected


def test_compatible_slice_takes_the_newest_minor_not_above_the_device():
    assert gpu_check.compatible_slice(["sm_80", "sm_90"], 8, 7) == "sm_80"
    assert gpu_check.compatible_slice(["sm_80", "sm_90"], 7, 5) == "none"


class RecordingLog:
    """Stands in for RunLog and keeps the results instead of writing a file."""

    opened: list["RecordingLog"] = []

    def __init__(self, *arguments) -> None:
        self.results = {}
        RecordingLog.opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> bool:
        return False


@pytest.fixture
def recorded(monkeypatch):
    RecordingLog.opened = []
    monkeypatch.setattr(gpu_check.runlog, "RunLog", RecordingLog)
    return RecordingLog.opened


def test_gpu_check_help_exits_without_running_or_logging(recorded, capsys):
    with pytest.raises(SystemExit) as stopped:
        gpu_check.main(["--help"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith("usage: python -m peddet_orin.gpu_check")
    assert recorded == []


def test_first_cuda_use_records_and_still_shows_the_torch_warning(monkeypatch):
    monkeypatch.setattr(torch.cuda, "init", lambda: warnings.warn(TORCH_WARNING, UserWarning, stacklevel=1))
    with pytest.warns(UserWarning, match="compute capability"):
        text = gpu_check.first_cuda_use()
    assert text == TORCH_WARNING


def test_first_cuda_use_without_a_warning_records_an_empty_string(monkeypatch):
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    assert gpu_check.first_cuda_use() == ""


def test_gpu_check_logs_the_warning_and_the_tf32_flags(monkeypatch, recorded):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: warnings.warn(TORCH_WARNING, UserWarning, stacklevel=1))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ARCH_LIST)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 7))
    monkeypatch.setattr(gpu_check, "matmul_difference", lambda generator: 0.0)
    monkeypatch.setattr(gpu_check, "nms_comparison", lambda generator: {"nms_identical": True})
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.warns(UserWarning, match="compute capability"):
        assert gpu_check.main([]) == 0
    results = recorded[0].results
    assert results["torch_warning"] == TORCH_WARNING
    assert results["cudnn_allow_tf32"] is False
    assert results["matmul_allow_tf32"] is False
    assert results["inferred_slice"] == "sm_87"
    assert results["gpu_torch_usable"] is True
