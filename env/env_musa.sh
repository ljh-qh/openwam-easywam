#!/usr/bin/env bash
# Source INSIDE the pinned lianjh-easywam-36096 container.
export EASYWAM_ROOT=/datapool/lianjh/easywam
export PATH="$EASYWAM_ROOT/envs/musa/bin:$PATH"
export PYTHONPATH="$EASYWAM_ROOT/code:$EASYWAM_ROOT/code/easywam/src:$EASYWAM_ROOT/code/vendor:$EASYWAM_ROOT/code/vendor/openwam-source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export DS_ACCELERATOR=musa
# This continuation is one machine; use GPU peer links without an IB path.
export MCCL_IB_DISABLE=1
export LD_LIBRARY_PATH="$EASYWAM_ROOT/envs/driver-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR="$EASYWAM_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$EASYWAM_ROOT/cache/triton"
