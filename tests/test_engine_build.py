# SPDX-License-Identifier: Apache-2.0
"""The engine build's input checks, CLI and fallback rule, on the CPU without TensorRT."""

import hashlib
import types
import warnings

import onnxruntime.quantization
import pytest

from peddet_orin import engine_build
from test_run_logs import make_log

IMAGE_COUNT = 4
# The flags an INT8 build reads back from its config.
INT8_FLAGS = ["FP16", "INT8", "TF32"]
# The text TensorRT 10.16 emits when IInt8EntropyCalibrator2 is used.
TENSORRT_DEPRECATION = "Use Deprecated in TensorRT 10.1. Superseded by explicit quantization. instead."


def image_name(index: int) -> str:
    return f"{index:012d}.jpg"


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    """A directory of IMAGE_COUNT files and a manifest listing their hashes."""
    directory = tmp_path / "calibration"
    directory.mkdir()
    lines = []
    for index in range(IMAGE_COUNT):
        content = f"image {index}".encode()
        (directory / image_name(index)).write_bytes(content)
        lines.append(f"{hashlib.sha256(content).hexdigest()}  {image_name(index)}\n")
    manifest = tmp_path / "calibration.sha256"
    manifest.write_text("".join(lines))
    monkeypatch.setattr(engine_build, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(engine_build, "EXPECTED_CALIBRATION_IMAGES", IMAGE_COUNT)
    return directory


def test_the_listed_calibration_set_is_accepted(calibration):
    images = engine_build.verify_calibration(calibration)
    assert [path.name for path in images] == [image_name(index) for index in range(IMAGE_COUNT)]


def test_one_missing_calibration_image_is_rejected(calibration):
    (calibration / image_name(2)).unlink()
    expected = f"1 of {IMAGE_COUNT} calibration images missing.*{image_name(2)}"
    with pytest.raises(engine_build.InputError, match=expected):
        engine_build.verify_calibration(calibration)


def test_a_calibration_image_with_the_wrong_hash_is_rejected(calibration):
    (calibration / image_name(1)).write_bytes(b"other bytes")
    with pytest.raises(engine_build.InputError, match=f"{image_name(1)} does not match its sha256"):
        engine_build.verify_calibration(calibration)


def test_an_extra_file_in_the_calibration_dir_is_rejected(calibration):
    (calibration / "000000999999.jpg").write_bytes(b"extra")
    with pytest.raises(engine_build.InputError, match="1 files .* not in the calibration manifest"):
        engine_build.verify_calibration(calibration)


def test_a_manifest_of_the_wrong_length_is_rejected(calibration):
    engine_build.MANIFEST_PATH.write_text(engine_build.MANIFEST_PATH.read_text().splitlines(keepends=True)[0])
    with pytest.raises(engine_build.InputError, match=f"lists 1 images, expected {IMAGE_COUNT}"):
        engine_build.verify_calibration(calibration)


def test_499_of_the_500_calibration_images_are_refused(monkeypatch, tmp_path):
    """At the real count of 500, one absent image stops the build."""
    directory = tmp_path / "calibration"
    directory.mkdir()
    lines = []
    for index in range(engine_build.EXPECTED_CALIBRATION_IMAGES):
        content = f"image {index}".encode()
        (directory / image_name(index)).write_bytes(content)
        lines.append(f"{hashlib.sha256(content).hexdigest()}  {image_name(index)}\n")
    manifest = tmp_path / "calibration.sha256"
    manifest.write_text("".join(lines))
    monkeypatch.setattr(engine_build, "MANIFEST_PATH", manifest)
    assert len(engine_build.verify_calibration(directory)) == 500
    (directory / image_name(123)).unlink()
    with pytest.raises(engine_build.InputError, match=f"1 of 500 calibration images missing.*{image_name(123)}"):
        engine_build.verify_calibration(directory)


def test_the_committed_manifest_lists_the_expected_count():
    assert len(engine_build.read_manifest(engine_build.MANIFEST_PATH)) == engine_build.EXPECTED_CALIBRATION_IMAGES


def test_a_missing_calibration_dir_is_rejected(calibration, tmp_path):
    with pytest.raises(engine_build.InputError, match="calibration directory not found"):
        engine_build.verify_calibration(tmp_path / "absent")


class FailingLog:
    def __init__(self, *arguments) -> None:
        raise AssertionError("no run log may be opened when an input is missing")


def test_build_with_a_missing_onnx_exits_2_with_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(engine_build.runlog, "RunLog", FailingLog)
    assert engine_build.main(["--precision", "fp16", "--onnx", str(tmp_path / "absent.onnx")]) == 2
    error = capsys.readouterr().err
    assert error.count("\n") == 1
    assert error.startswith("python -m peddet_orin.engine_build: error: ONNX file not found:")
    assert "absent.onnx" in error


def test_int8_build_with_a_bad_calibration_set_exits_2(calibration, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(engine_build.runlog, "RunLog", FailingLog)
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    (calibration / image_name(0)).write_bytes(b"tampered")
    arguments = ["--precision", "int8", "--onnx", str(onnx_path), "--calibration-dir", str(calibration)]
    assert engine_build.main(arguments) == 2
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "does not match its sha256" in error
    assert error.startswith("python -m peddet_orin.engine_build: error: ")


def test_the_fallback_is_not_taken_when_the_onnx_is_missing(calibration, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(engine_build, "_build_int8_entropy", lambda *arguments: called.append("entropy"))
    monkeypatch.setattr(engine_build, "_build_int8_qdq", lambda *arguments: called.append("qdq"))
    images = engine_build.verify_calibration(calibration)
    with pytest.raises(engine_build.InputError, match="ONNX file not found"):
        engine_build.build_int8(tmp_path / "absent.onnx", tmp_path / "int8.engine", images)
    assert called == []


def test_a_bug_in_the_entropy_build_is_not_sent_to_the_fallback(calibration, tmp_path, monkeypatch):
    def broken(*arguments):
        raise TypeError("a programming error")

    called = []
    monkeypatch.setattr(engine_build, "_build_int8_entropy", broken)
    monkeypatch.setattr(engine_build, "_build_int8_qdq", lambda *arguments: called.append("qdq"))
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    with pytest.raises(TypeError, match="a programming error"):
        engine_build.build_int8(onnx_path, tmp_path / "int8.engine", engine_build.verify_calibration(calibration))
    assert called == []


def test_a_tensorrt_build_failure_takes_the_qdq_fallback(calibration, tmp_path, monkeypatch):
    def no_engine(*arguments):
        raise engine_build.TensorRtBuildError("TensorRT returned no engine")

    called = []
    monkeypatch.setattr(engine_build, "_build_int8_entropy", no_engine)
    monkeypatch.setattr(engine_build, "_build_int8_qdq", lambda *arguments: called.append("qdq"))
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    images = engine_build.verify_calibration(calibration)
    route, failure, _ = engine_build.build_int8(onnx_path, tmp_path / "int8.engine", images)
    assert (route, called) == (engine_build.PATH_QDQ, ["qdq"])
    assert failure == "TensorRtBuildError: TensorRT returned no engine"


def test_the_calibrator_deprecation_filter_matches_tensorrts_text():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=engine_build.CALIBRATOR_DEPRECATION_PATTERN)
        warnings.warn(TENSORRT_DEPRECATION, DeprecationWarning, stacklevel=1)
        warnings.warn("Use Deprecated in TensorRT 10.1. Some other API. instead.", DeprecationWarning, stacklevel=1)
    assert [str(warning.message) for warning in caught] == ["Use Deprecated in TensorRT 10.1. Some other API. instead."]


def test_engine_build_help_names_its_module_and_every_option(capsys):
    with pytest.raises(SystemExit) as stopped:
        engine_build.parse_arguments(["--help"])
    assert stopped.value.code == 0
    text = capsys.readouterr().out
    assert text.startswith("usage: python -m peddet_orin.engine_build")
    for option in ("--precision", "--onnx", "--output", "--calibration-dir"):
        assert option in text
    assert "artifacts/<precision>.engine" in text


def test_output_and_calibration_dir_options_are_parsed(tmp_path):
    args = engine_build.parse_arguments(
        ["--precision", "int8", "--output", str(tmp_path / "x.engine"), "--calibration-dir", str(tmp_path)]
    )
    assert (args.output, args.calibration_dir) == (tmp_path / "x.engine", tmp_path)
    assert engine_build.parse_arguments(["--precision", "fp16"]).output is None


class RecordingLog:
    opened: list = []

    def __init__(self, kind, precision, seed, iterations) -> None:
        self.results = {}
        self.iterations = iterations
        RecordingLog.opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> bool:
        return False


def test_int8_build_logs_the_manifest_hash_and_route(calibration, tmp_path, monkeypatch):
    RecordingLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingLog)
    output = tmp_path / "out" / "int8.engine"

    def fake_entropy(onnx_path, engine, images):
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        return INT8_FLAGS

    monkeypatch.setattr(engine_build, "_build_int8_entropy", fake_entropy)
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    arguments = [
        "--precision",
        "int8",
        "--onnx",
        str(onnx_path),
        "--output",
        str(output),
        "--calibration-dir",
        str(calibration),
    ]
    assert engine_build.main(arguments) == 0
    log = RecordingLog.opened[0]
    assert log.iterations == {"calibration_images": IMAGE_COUNT}
    assert log.results["int8_path"] == engine_build.PATH_ENTROPY
    assert log.results["entropy_failure"] == ""
    assert log.results["engine_bytes"] == len(b"engine")
    digest = hashlib.sha256(engine_build.MANIFEST_PATH.read_bytes()).hexdigest()
    assert log.results["calibration_manifest_sha256"] == digest
    assert log.results["engine_sha256"] == hashlib.sha256(b"engine").hexdigest()


def test_the_qdq_fallback_quantizes_symmetrically_without_biases(tmp_path, monkeypatch):
    # TensorRT cannot parse a DequantizeLinear of an int32 bias, nor a non-zero zero point.
    seen = {}
    monkeypatch.setattr(onnxruntime.quantization, "quantize_static", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(engine_build, "_build", lambda *a, **k: None)
    engine_build._build_int8_qdq(tmp_path / "in.onnx", tmp_path / "out.engine", [])
    assert seen["extra_options"] == {"QuantizeBias": False, "ActivationSymmetric": True, "WeightSymmetric": True}


class FakeConfig:
    """Stands in for a TensorRT builder config: flags are set, cleared and read back, and TF32 starts on, as in
    TensorRT. ``ignored`` names flags whose set_flag does nothing, to plant a config that does not take a flag."""

    def __init__(self, ignored: frozenset[str] = frozenset()) -> None:
        self.state: set[str] = {"TF32"}
        self.ignored = ignored
        self.int8_calibrator = None

    def set_flag(self, flag) -> None:
        if flag not in self.ignored:
            self.state.add(flag)

    def clear_flag(self, flag) -> None:
        self.state.discard(flag)

    def get_flag(self, flag) -> bool:
        return flag in self.state


# Stands in for the tensorrt module as configure sees it: each BuilderFlag is its own name. This TensorRT has no FP8.
FAKE_TENSORRT = types.SimpleNamespace(
    BuilderFlag=types.SimpleNamespace(FP16="FP16", INT8="INT8", TF32="TF32", BF16="BF16")
)


class FakeCalibrator:
    released = False

    def release(self) -> None:
        self.released = True


@pytest.fixture
def configured(monkeypatch):
    """Replace the TensorRT build with one that applies configure to a FakeConfig and returns the flags read back
    from it, as _build does; return the configs made. Set ``configs.ignored`` to plant a config that drops flags."""
    configs = types.SimpleNamespace(made=[], ignored=frozenset())

    def fake_build(onnx_path, output, configure) -> list[str]:
        config = FakeConfig(configs.ignored)
        config.state.add("BF16")  # A flag the build must clear.
        configure(config, FAKE_TENSORRT)
        configs.made.append(config)
        return engine_build.flags_set(config, FAKE_TENSORRT)

    monkeypatch.setattr(engine_build, "_build", fake_build)
    monkeypatch.setattr(engine_build, "_make_calibrator", lambda images, cache_path: FakeCalibrator())
    monkeypatch.setattr(onnxruntime.quantization, "quantize_static", lambda *a, **k: None)
    return configs


def test_flags_are_read_back_from_the_config_and_absent_flags_skipped():
    config = FakeConfig()
    config.state |= {"INT8", "FP8"}
    assert engine_build.flags_set(config, FAKE_TENSORRT) == ["INT8", "TF32"]


def test_fp16_has_fp16_and_tensorrts_default_tf32(configured, tmp_path):
    assert engine_build.build_fp16(tmp_path / "in.onnx", tmp_path / "fp16.engine") == ["FP16", "TF32"]
    assert configured.made[0].state == {"FP16", "TF32"}


def test_tensorrt_fp32_has_no_precision_flag(configured, tmp_path):
    assert engine_build.build_fp32_trt(tmp_path / "in.onnx", tmp_path / "fp32-trt.engine") == []
    assert configured.made[0].state == set()


def test_the_int8_entropy_build_has_fp16_int8_and_tf32(configured, tmp_path):
    flags = engine_build._build_int8_entropy(tmp_path / "in.onnx", tmp_path / "int8.engine", [])
    assert flags == ["FP16", "INT8", "TF32"]
    assert (
        isinstance(configured.made[0].int8_calibrator, FakeCalibrator) and configured.made[0].int8_calibrator.released
    )


def test_the_int8_qdq_build_has_fp16_int8_and_tf32(configured, tmp_path):
    assert engine_build._build_int8_qdq(tmp_path / "in.onnx", tmp_path / "int8.engine", []) == ["FP16", "INT8", "TF32"]


def run_main(precision, calibration, tmp_path, monkeypatch) -> dict:
    RecordingLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingLog)
    output = tmp_path / "out" / f"{precision}.engine"
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b"engine")
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    arguments = ["--precision", precision, "--onnx", str(onnx_path), "--output", str(output)]
    assert engine_build.main([*arguments, "--calibration-dir", str(calibration)]) == 0
    return RecordingLog.opened[0].results


@pytest.mark.parametrize("precision", engine_build.PRECISIONS)
def test_the_build_log_records_the_flags_read_back(precision, configured, calibration, tmp_path, monkeypatch):
    results = run_main(precision, calibration, tmp_path, monkeypatch)
    assert results["builder_flags"] == engine_build.flags_set(configured.made[0], FAKE_TENSORRT)
    assert results["builder_flags"] == engine_build.runlog.BUILDER_FLAG_NAMES[precision]
    assert "cleared_builder_flags" not in results


def test_a_config_that_drops_int8_gives_a_refused_build_log(configured, calibration, tmp_path, monkeypatch):
    configured.ignored = frozenset({"INT8"})
    results = run_main("int8", calibration, tmp_path, monkeypatch)
    assert results["builder_flags"] == ["FP16", "TF32"]
    problems = engine_build.runlog.validate_log(make_log("build", "int8", results=results))
    assert "bad result: build builder_flags must be ['FP16', 'INT8', 'TF32'] for int8" in problems


def run_int8_build(calibration, tmp_path, monkeypatch, entropy) -> dict:
    """Run main for INT8 with ``entropy`` standing in for the entropy build; return the build log's results."""
    RecordingLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingLog)
    monkeypatch.setattr(engine_build, "_build_int8_entropy", entropy)
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    output = tmp_path / "out" / "int8.engine"
    arguments = ["--precision", "int8", "--onnx", str(onnx_path), "--output", str(output)]
    assert engine_build.main([*arguments, "--calibration-dir", str(calibration)]) == 0
    return RecordingLog.opened[0].results


def test_the_int8_build_log_records_the_engine_and_cache_hashes(calibration, tmp_path, monkeypatch):
    def entropy_writes_both(onnx_path, engine, images):
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        (engine.parent / engine_build.CALIBRATION_CACHE_NAME).write_bytes(b"cache")
        return INT8_FLAGS

    results = run_int8_build(calibration, tmp_path, monkeypatch, entropy_writes_both)
    assert results["engine_sha256"] == hashlib.sha256(b"engine").hexdigest()
    assert results["calibration_cache_sha256"] == hashlib.sha256(b"cache").hexdigest()


def test_an_int8_build_that_writes_no_cache_records_none(calibration, tmp_path, monkeypatch):
    def entropy_writes_engine(onnx_path, engine, images):
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        return INT8_FLAGS

    assert run_int8_build(calibration, tmp_path, monkeypatch, entropy_writes_engine)["calibration_cache_sha256"] == "-"


def test_the_qdq_fallback_records_no_calibration_cache(calibration, tmp_path, monkeypatch):
    """A cache the failed entropy attempt left is not the engine's: the log records none."""

    def entropy_fails(onnx_path, engine, images):
        engine.parent.mkdir(parents=True, exist_ok=True)
        (engine.parent / engine_build.CALIBRATION_CACHE_NAME).write_bytes(b"cache")
        raise engine_build.TensorRtBuildError("TensorRT returned no engine")

    def qdq_writes_engine(onnx_path, engine, images):
        engine.write_bytes(b"qdq engine")
        return INT8_FLAGS

    monkeypatch.setattr(engine_build, "_build_int8_qdq", qdq_writes_engine)
    results = run_int8_build(calibration, tmp_path, monkeypatch, entropy_fails)
    # The cache the failed attempt wrote is removed before the fallback builds.
    assert not (tmp_path / "out" / engine_build.CALIBRATION_CACHE_NAME).exists()
    assert results["int8_path"] == engine_build.PATH_QDQ
    assert results["engine_sha256"] == hashlib.sha256(b"qdq engine").hexdigest()
    assert results["calibration_cache_sha256"] == "-"


def test_the_entropy_build_removes_an_earlier_builds_cache_first(configured, tmp_path):
    stale = tmp_path / engine_build.CALIBRATION_CACHE_NAME
    stale.write_bytes(b"an earlier build's cache")
    engine_build._build_int8_entropy(tmp_path / "in.onnx", tmp_path / "int8.engine", [])
    assert not stale.exists()


def test_the_fp16_build_log_records_the_engine_hash_and_no_cache(configured, tmp_path, monkeypatch):
    RecordingLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingLog)
    output = tmp_path / "fp16.engine"
    output.write_bytes(b"fp16 engine")
    (tmp_path / engine_build.CALIBRATION_CACHE_NAME).write_bytes(b"cache")
    onnx_path = tmp_path / "network.onnx"
    onnx_path.write_bytes(b"onnx")
    assert engine_build.main(["--precision", "fp16", "--onnx", str(onnx_path), "--output", str(output)]) == 0
    results = RecordingLog.opened[0].results
    assert results["engine_sha256"] == hashlib.sha256(b"fp16 engine").hexdigest()
    assert results["calibration_cache_sha256"] == "-"
