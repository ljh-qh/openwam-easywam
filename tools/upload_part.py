"""Upload one 1 GiB part with 5 retries (used for concurrency tests and the parallel uploader)."""
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = 'ljhqh24/openwam'
PREFIX = 'checkpoints/top_state_5ep_33050'
PARTS = Path('/datapool/lianjh/openwam_top/work/parts_33050')
TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()


def main():
    part = sys.argv[1]
    src = PARTS / part
    dst = f'{PREFIX}/parts/{part}'
    api = HubApi()
    api.login(TOKEN)
    for attempt in range(1, 6):
        try:
            start = time.time()
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
            print(f'DONE {part} {src.stat().st_size} {time.time() - start:.1f}s', flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f'RETRY {part} {attempt} {type(exc).__name__}: {str(exc)[:160]}', flush=True)
            time.sleep(min(30 * attempt, 120))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
