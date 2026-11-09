"""Split checkpoint_step_33050 into 1 GiB parts with a sha256 manifest."""
import hashlib
import json
from pathlib import Path

SRC = Path('/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56/checkpoint_step_33050.safetensors')
OUT = Path('/datapool/lianjh/openwam_top/work/parts_33050')
PART = 1024 ** 3
REPO = 'ljhqh24/openwam'
PREFIX = 'checkpoints/top_state_5ep_33050'


def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} already exists; remove it first')
    OUT.mkdir(parents=True)
    whole = hashlib.sha256()
    total = 0
    parts = []
    with open(SRC, 'rb') as f:
        index = 0
        while True:
            data = f.read(PART)
            if not data:
                break
            whole.update(data)
            total += len(data)
            name = f'part_{index:02d}'
            (OUT / name).write_bytes(data)
            parts.append({'name': name, 'size': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
            print(f'wrote {name} {len(data)} bytes', flush=True)
            index += 1
    manifest = {
        'checkpoint': SRC.name,
        'total_size': total,
        'sha256': whole.hexdigest(),
        'source': str(SRC),
        'repo': REPO,
        'prefix': PREFIX,
        'parts': parts,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('MANIFEST_OK', total, whole.hexdigest(), len(parts), flush=True)


if __name__ == '__main__':
    main()
