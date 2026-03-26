# Changelog

All notable changes to this project will be documented in this file.

## [2026-03-20] - Fix Flux Nunchaku GPU OOM

### Fixed
- Added `cpu_offload="auto"` and unified parameters for `NunchakuFluxDiTLoader` to enable CPU/RAM fallback and prevent 1 GB tensor allocation failures on GPU.
- Detailed report in [2026-03-20 - fix-nunchaku-flux-offload.md](./changelog-details/2026-03-20%20-%20fix-nunchaku-flux-offload.md).
