from pathlib import Path

import pyarrow.parquet as pq
from safetensors import safe_open

checks = {
    'top tasks.parquet': '/datapool/lianjh/openwam_top/data/openwam_eef80/meta/tasks.parquet',
    'top data parquet': '/datapool/lianjh/openwam_top/data/openwam_eef80/data/chunk-000/file-000.parquet',
    'badminton episodes parquet': '/datapool/lianjh/openwam/data/openwam_eef80/meta/episodes/chunk-000/file-000.parquet',
}
for name, path in checks.items():
    table = pq.read_table(path).slice(0, 1)
    print('PARQUET_OK', name, '->', Path(path).resolve(), 'rows', table.num_rows, 'cols', len(table.column_names))

st = '/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56/checkpoint_step_33050.safetensors'
with safe_open(st, framework='pt') as f:
    keys = list(f.keys())
print('SAFETENSORS_OK', Path(st).resolve(), 'tensors', len(keys), keys[:3])
