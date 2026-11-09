#!/usr/bin/env bash
set -euo pipefail
ROOT=/datapool/lianjh/openwam
source "$ROOT/scripts/env_openwam.sh"
exec 9>"$ROOT/logs/bottle_training.lock"
flock -n 9 || { echo 'Hybrid training already running'; exit 1; }
python - <<'PY'
import torch,torch_musa
assert torch.musa.device_count()==8
for i in range(8):
    free,total=torch.musa.mem_get_info(i)
    assert free>70*1024**3,(i,free)
print('MUSA_8_GPU_MEMORY_PREFLIGHT_PASS',flush=True)
PY
cd "$ROOT/code/source"
exec python -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29631 "$ROOT/scripts/train_bottle.py" "$@"
