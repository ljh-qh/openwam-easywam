"""Normalize bottle episode directory names to episode_XXXX using manifest.episode."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path('/datapool/lianjh/openwam/data/bottle/raw/episode_bottle')
CLEAN = re.compile(r'episode_(\d{4})')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    renamed, duplicates, kept = [], [], 0
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        m = CLEAN.fullmatch(d.name)
        if m:
            kept += 1
            continue
        manifest = json.loads((d / 'manifest.json').read_text())
        ep = int(manifest['episode'])
        target = ROOT / f'episode_{ep:04d}'
        if target.exists():
            h5 = d / 'robot_states.h5'
            same = (h5.stat().st_size == (target / 'robot_states.h5').stat().st_size
                    and sha256(h5) == sha256(target / 'robot_states.h5'))
            dup_dir = ROOT / '.duplicates'
            dup_dir.mkdir(exist_ok=True)
            dest = dup_dir / d.name
            shutil.move(str(d), str(dest))
            duplicates.append((d.name, target.name, same))
            print(f'DUPLICATE {d.name!r} -> {dest} (identical_h5={same})', flush=True)
        else:
            d.rename(target)
            renamed.append((d.name, target.name))
            print(f'RENAMED {d.name!r} -> {target.name}', flush=True)

    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and CLEAN.fullmatch(p.name))
    expected = [f'episode_{i:04d}' for i in range(1, len(names) + 1)]
    gaps = sorted(set(expected) - set(names))
    extras = sorted(set(names) - set(expected))
    print(json.dumps({'kept': kept, 'renamed': renamed, 'duplicates': duplicates,
                      'total_clean_dirs': len(names), 'gaps': gaps, 'extras': extras},
                     indent=2))
    (ROOT.parent / 'name_cleanup_report.json').write_text(json.dumps(
        {'kept': kept, 'renamed': renamed, 'duplicates': duplicates,
         'total_clean_dirs': len(names), 'gaps': gaps, 'extras': extras}, indent=2) + '\n')


if __name__ == '__main__':
    main()
