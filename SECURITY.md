# Security policy

## Reporting a vulnerability

Please report a vulnerability privately, through GitHub's private vulnerability reporting: open the repository's
**Security** tab and choose **Report a vulnerability**
(https://github.com/ahmedhassan-eng/peddet-orin/security/advisories/new).

Do not open a public issue for a vulnerability. Reports are handled, and any fix is announced, as a GitHub
security advisory.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## In scope

- Code in this repository: `src/peddet_orin/`, `scripts/` and `tests/`.
- `scripts/fetch_coco.py`: what it downloads (over HTTPS; the final URL a download reaches after any redirect is checked too, and a non-HTTPS one is refused before anything is written), how it checks the MD5 and sha256 digests, and where it
  extracts archives. Its calibration subcommand refuses to run without the calibration manifest unless `--write-manifest` is given.
- `src/peddet_orin/model.py`: the pretrained weights load only after their full sha256 matches, and only as tensors.
- The run logs' path hygiene: `src/peddet_orin/runlog.py` replaces the absolute paths its pattern matches, outside
  its allowlist, with `<path>` before it writes a log. The pattern misses a path that follows a word character, `.`,
  `-` or `~`.
- The network guard in `tests/network_guard.py`, which refuses internet connections from the test process; it does
  not guard a subprocess a test starts or the C-level `_socket` module.
- The CI workflow in `.github/workflows/ci.yml`, which audits both pin files with `pip-audit --strict`.

## Out of scope

- Vulnerabilities in PyTorch, torchvision, ONNX, ONNX Runtime, TensorRT, CUDA or JetPack themselves. Report those to
  their own projects.
- Loading model files from untrusted sources. A TensorRT engine or a PyTorch checkpoint can run code or crash the
  process when loaded; build engines yourself and load only files you trust.
- The COCO dataset and the content of the pretrained torchvision weights, which this project downloads but does not
  ship.
- Benchmark numbers that differ on other boards, power modes or software versions. Open an issue for those.
