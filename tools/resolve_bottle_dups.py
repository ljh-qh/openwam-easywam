"""Keep the longer recording when two dirs claim the same bottle episode id."""
import json
from pathlib import Path

ROOT = Path('/datapool/lianjh/openwam/data/bottle/raw/episode_bottle')
DUP = ROOT / '.duplicates'


def main_frames(d):
    m = json.loads((d / 'manifest.json').read_text())
    return int((m.get('cameras') or {}).get('main', {}).get('frames', -1))


def main():
    decisions = []
    for d in sorted(DUP.iterdir()):
        if not d.is_dir() or d.name.startswith('kept_'):
            continue
        m = json.loads((d / 'manifest.json').read_text())
        ep = int(m['episode'])
        target = ROOT / f'episode_{ep:04d}'
        if not target.exists():
            d.rename(target)
            decisions.append({'duplicate': d.name, 'kept': target.name, 'action': 'restored'})
            print('RESTORED', d.name, '->', target.name, flush=True)
            continue
        dup_frames, kept_frames = main_frames(d), main_frames(target)
        if dup_frames > kept_frames:
            parked = DUP / f'kept_{target.name}'
            target.rename(parked)
            d.rename(target)
            decisions.append({'duplicate': d.name, 'kept': target.name, 'action': 'swapped',
                              'kept_frames': dup_frames, 'parked_frames': kept_frames})
            print('SWAPPED', d.name, '->', target.name, 'frames', dup_frames, '>', kept_frames, flush=True)
        else:
            decisions.append({'duplicate': d.name, 'kept': target.name, 'action': 'kept_existing',
                              'kept_frames': kept_frames, 'dropped_frames': dup_frames})
            print('KEPT', target.name, kept_frames, '>=', dup_frames, flush=True)
    (ROOT.parent / 'duplicate_resolution.json').write_text(json.dumps({'decisions': decisions}, indent=2) + '\n')
    print('DECISIONS', len(decisions))


if __name__ == '__main__':
    main()
