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
- `scripts/fetch_coco.py`: what it downloads, how it checks the MD5 and sha256 digests, and where it extracts archives.
- The run logs' path hygiene: `src/peddet_orin/runlog.py` must not write absolute paths outside its allowlist.
- The network guard in `tests/network_guard.py`, which keeps the tests offline.
- The CI workflow in `.github/workflows/ci.yml`.

## Out of scope

- Vulnerabilities in PyTorch, torchvision, ONNX, ONNX Runtime, TensorRT, CUDA or JetPack themselves. Report those to
  their own projects.
- Loading model files from untrusted sources. A TensorRT engine or a PyTorch checkpoint can run code or crash the
  process when loaded; build engines yourself and load only files you trust.
- The COCO dataset and the pretrained torchvision weights, which this project downloads but does not ship.
- Benchmark numbers that differ on other boards, power modes or software versions. Open an issue for those.
