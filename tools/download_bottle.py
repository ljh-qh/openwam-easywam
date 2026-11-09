"""Download bottle raw recordings (dataset repo) into the merged project."""
from pathlib import Path

from modelscope import snapshot_download

DEST = Path('/datapool/lianjh/openwam/data/bottle/raw')
DEST.mkdir(parents=True, exist_ok=True)
errors = []
for repo_type in ('dataset', 'model'):
    try:
        path = snapshot_download('jiawangli/physwam', repo_type=repo_type,
                                 allow_patterns=['episode_bottle/**'],
                                 local_dir=str(DEST))
        print('DOWNLOAD_DONE', repo_type, path, flush=True)
        break
    except Exception as exc:  # noqa: BLE001
        errors.append(f'{repo_type}: {type(exc).__name__}: {str(exc)[:200]}')
        print('DOWNLOAD_TRY_FAILED', errors[-1], flush=True)
else:
    raise SystemExit('ALL_REPO_TYPES_FAILED ' + ' | '.join(errors))
episodes = sorted((DEST / 'episode_bottle').iterdir())
print('EPISODE_DIRS', len(episodes), flush=True)
