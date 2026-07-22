# AMD ROCm Migration

## Scope

Clone source from GitHub on the target server, then copy only runtime state that
cannot be reconstructed. Do not copy `.venv`, GPU-specific PyTorch wheels,
upstream TensorBoard logs, or the 46 GB local W&B media directory.

## Audit Target Hardware

Run these commands before selecting a PyTorch build:

```bash
cat /etc/os-release
uname -r
lspci -nn | grep -Ei 'amd|display|vga'
rocminfo | grep -E 'Name:|Marketing Name:|gfx'
rocm-smi --showproductname --showmeminfo vram
```

The exact GPU architecture, operating system, Python version, and installed
ROCm driver must match an AMD-supported combination. Do not guess the wheel
URL. Select it from the
[AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
and [AMD PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
PyTorch ROCm intentionally retains `torch.cuda` APIs and `cuda` device strings,
as documented in [HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html).

## Prepare Repository

```bash
git clone --recurse-submodules https://github.com/TaqiyudinMiftah/risetv2.git ~/riset/risetv2
cd ~/riset/risetv2
scripts/bootstrap_rocm_env.sh <compatible-pytorch-rocm-index-url>
```

The bootstrap creates `.venv`, installs the selected ROCm build before the
general dependencies, and runs a GPU matrix-multiplication check.

### Current Target Profile

The target `labkc2-ThinkCentre-M75t-Gen-5` has ROCm 7.2.1, an RX 6600 LE
(`gfx1032`, 8 GB), and an integrated `gfx1103` GPU with only 2 GB. Use device 0
only. The RX 6600 is outside the current officially supported Radeon list. The
following compatibility setup was verified with matrix multiplication and a
Conv2d/BatchNorm backward pass, but must be reported as an environment
workaround rather than official hardware support:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
TORCH_SPEC='torch==2.5.1' \
TORCHVISION_SPEC='torchvision==0.20.1' \
scripts/bootstrap_rocm_env.sh https://download.pytorch.org/whl/rocm6.2
```

## Copy Runtime State

From the source machine, after SSH connectivity is available:

```bash
scripts/migrate_runtime_state.sh taqiyudinmiftah@100.110.19.16 riset/risetv2
```

This transfers `CAER-S/`, protocol artifacts, detector annotations, compact
checkpoints, paper assets, and only `model_best.pth`/`config.json` from upstream
runs. It excludes reproducible logs, `.venv`, caches, and W&B media.

## Verify and Train

Use one device first. Increase `N_GPU` only after the target inventory confirms
multiple compatible AMD GPUs.

```bash
.venv/bin/python scripts/check_accelerator.py --require-backend rocm --min-devices 1
.venv/bin/python run_caer_clean.py train \
  --config configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0 --n-gpu 1 --wandb-mode offline --smoke-only
HSA_OVERRIDE_GFX_VERSION=10.3.0 DEVICE_IDS=0 N_GPU=1 scripts/launch_clean_tmux.sh
```

Monitor the detached run with `tmux attach -t caer-clean-s42` or tail the path
printed by the launcher. Test data remains excluded; checkpoint selection uses
validation macro F1.
