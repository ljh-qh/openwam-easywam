"""Upload the two remaining small files with tolerant remote-size verification."""
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = 'ljhqh24/easywam'
TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()
ITEMS = [
    (Path('/datapool/lianjh/easywam/official/badminton/context/top_context.pt'), 'top_5ep/top_context.pt'),
    (Path('/datapool/lianjh/openwam_top/work/parts_easywam_4131/manifest.json'), 'top_5ep/upload_manifest.json'),
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def remote_size(api, path):
    try:
        for item in api.get_model_files(REPO, recursive=True):
            p = str(item.get('Path', '')) if isinstance(item, dict) else str(getattr(item, 'Path', ''))
            if p == path:
                return item.get('Size') if isinstance(item, dict) else getattr(item, 'Size', None)
    except Exception as exc:  # noqa: BLE001
        log(f'LIST_ERR {type(exc).__name__} {str(exc)[:100]}')
    return None


def main():
    api = HubApi()
    api.login(TOKEN)
    for src, dst in ITEMS:
        want = src.stat().st_size
        for attempt in range(1, 4):
            log(f'UPLOAD {dst} {want} attempt={attempt}')
            try:
                api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
            except Exception as exc:  # noqa: BLE001
                log(f'RETRY {dst} {type(exc).__name__}: {str(exc)[:120]}')
                time.sleep(20)
                continue
            for _ in range(12):
                got = remote_size(api, dst)
                if got == want:
                    log(f'DONE {dst} {want}')
                    break
                time.sleep(10)
            else:
                log(f'STILL_MISMATCH {dst} remote={got} want={want}')
                continue
            break
        else:
            raise SystemExit(f'failed {dst}')
    rows = []
    for item in api.get_model_files(REPO, recursive=True):
        p = str(item.get('Path', '')) if isinstance(item, dict) else str(getattr(item, 'Path', ''))
        if p.startswith('top_5ep/'):
            rows.append((p, item.get('Size') if isinstance(item, dict) else getattr(item, 'Size', None)))
    for p, s in sorted(rows):
        print('REMOTE', p, s)
    log('SMALL_UPLOAD_DONE')


if __name__ == '__main__':
    main()
