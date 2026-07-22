# AMD ROCm Migration Report

## Target

- Host: `labkc2-ThinkCentre-M75t-Gen-5`
- OS: Ubuntu 22.04.5, kernel 6.8.0-107
- CPU/RAM: Ryzen 7 8700G, 16 threads, 60 GiB RAM
- Training GPU: RX 6600 LE, `gfx1032`, 8176 MiB VRAM
- Excluded GPU: integrated `gfx1103`, 2048 MiB VRAM
- Driver/ROCm: amdgpu 6.16.13, system ROCm 7.2.1

The RX 6600 is not in the current officially supported Radeon list. PyTorch
2.5.1+rocm6.2 runs successfully with
`HSA_OVERRIDE_GFX_VERSION=10.3.0`; this is recorded as a compatibility
workaround, not official hardware support.

## Migrated State

The clean checkout is `/home/taqiyudinmiftah/riset/risetv2`. The existing
CAER-S files were reused through a non-destructive class-normalizing symlink
layout. The normalized filename/size inventory hash is
`40437886f4927e0584ebc23a64784204e32c4a46a10f60a9a74c54cb228874cb`.
The content-disjoint manifest SHA-256 remains
`f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.

Protocol artifacts, detector annotations, compact checkpoints, paper assets,
and upstream best checkpoints/configs were copied. Reproducible upstream logs,
W&B media, caches, and the NVIDIA virtual environment were excluded.

## Validation

- 34 unit tests passed on the target server.
- ROCm matrix multiplication passed on device 0.
- CAER-Net smoke forward passed with face `[128, 3, 96, 96]`, context
  `[128, 3, 112, 112]`, and logits `[128, 7]`.
- The smoke path reported `test_accessed: false`.

Training started in tmux session `caer-clean-s42-rocm` with run ID
`caernet__clean_inrepo__seed42__20260722_043253`. It uses one RX 6600,
global batch 128, seed 42, validation macro F1 checkpoint selection, and W&B
offline mode. The test split remains locked.
