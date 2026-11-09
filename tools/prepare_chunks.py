"""Split a large file into fixed-size chunks with a sha256 manifest (generic)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--part-size', type=int, default=1024 ** 3)
    parser.add_argument('--part-prefix', default='part_')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} already exists; remove it first')
    out.mkdir(parents=True)
    whole = hashlib.sha256()
    total = 0
    parts = []
    with open(src, 'rb') as f:
        index = 0
        while True:
            data = f.read(args.part_size)
            if not data:
                break
            whole.update(data)
            total += len(data)
            name = f'{args.part_prefix}{index:02d}'
            (out / name).write_bytes(data)
            parts.append({'name': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            print(f'wrote {name} {len(data)}', flush=True)
            index += 1
    manifest = {
        'checkpoint': src.name,
        'total_size': total,
        'sha256': whole.hexdigest(),
        'source': str(src),
        'repo': args.repo,
        'prefix': args.prefix,
        'parts': parts,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('MANIFEST_OK', total, whole.hexdigest(), len(parts), flush=True)


if __name__ == '__main__':
    main()
