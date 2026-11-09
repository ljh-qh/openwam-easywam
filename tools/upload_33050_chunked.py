"""Chunked upload of checkpoint_step_33050 to ModelScope (per-part retry/skip)."""
import json
import socket
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = 'ljhqh24/openwam'
PREFIX = 'checkpoints/top_state_5ep_33050'
PARTS = Path('/datapool/lianjh/openwam_top/work/parts_33050')
RUN = Path('/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56')
TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()
LOG = Path('/datapool/lianjh/openwam_top/logs/upload_33050_chunked.log')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)


def remote_sizes(api):
    files = api.get_model_files(REPO, recursive=True)
    out = {}
    for item in files:
        if isinstance(item, dict):
            path = item.get('Path') or item.get('path')
            size = item.get('Size', item.get('size'))
        else:
            path = getattr(item, 'Path', None) or getattr(item, 'path', None)
            size = getattr(item, 'Size', None) or getattr(item, 'size', None)
        if path:
            out[str(path)] = int(size) if size is not None else None
    return out


def upload_one(api, src, dst, expected_size, retries=5):
    for attempt in range(1, retries + 1):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
        except Exception as exc:  # noqa: BLE001 - network errors are expected
            log(f'RETRY {dst} attempt={attempt} {type(exc).__name__}: {str(exc)[:160]}')
            time.sleep(min(30 * attempt, 120))
            continue
        try:
            names = remote_sizes(api)
            got = names.get(dst)
        except Exception as exc:  # noqa: BLE001
            log(f'VERIFY_RETRY {dst} {type(exc).__name__}: {str(exc)[:120]}')
            got = None
        if got == expected_size:
            log(f'DONE {dst} {expected_size}')
            return True
        log(f'SIZE_MISMATCH {dst} remote={got} expected={expected_size} attempt={attempt}')
        time.sleep(min(30 * attempt, 120))
    return False


def main():
    manifest = json.loads((PARTS / 'manifest.json').read_text())
    api = HubApi()
    api.login(TOKEN)
    log(f'LOGIN_OK {REPO}')
    existing = {}
    try:
        existing = remote_sizes(api)
        log(f'REMOTE_FILES {len(existing)}')
    except Exception as exc:  # noqa: BLE001
        log(f'REMOTE_LIST_FAILED {type(exc).__name__}: {str(exc)[:160]}')

    pending = []
    for part in manifest['parts']:
        dst = f"{PREFIX}/parts/{part['name']}"
        src = PARTS / part['name']
        if existing.get(dst) == part['size']:
            log(f'SKIP {dst} (already {part["size"]})')
            continue
        pending.append((src, dst, part['size']))
    log(f'PARTS_TODO {len(pending)}/{len(manifest["parts"])}')

    for index, (src, dst, size) in enumerate(pending, 1):
        log(f'UPLOAD [{index}/{len(pending)}] {dst} {size}')
        if not upload_one(api, src, dst, size):
            log(f'FAILED {dst}')
            raise SystemExit(2)

    extras = [
        (RUN / 'config.yaml', f'{PREFIX}/config.yaml'),
        (RUN / 'normalization_stats.npy', f'{PREFIX}/normalization_stats.npy'),
    ]
    for path in sorted((RUN / 'tokenizer').rglob('*')):
        if path.is_file():
            extras.append((path, f'{PREFIX}/tokenizer/{path.relative_to(RUN / "tokenizer")}'))
    for src, dst in extras:
        size = src.stat().st_size
        if existing.get(dst) == size:
            log(f'SKIP {dst} (already {size})')
            continue
        log(f'UPLOAD {dst} {size}')
        if not upload_one(api, src, dst, size):
            log(f'FAILED {dst}')
            raise SystemExit(3)

    manifest_src = PARTS / 'manifest.json'
    dst = f'{PREFIX}/upload_manifest.json'
    if not upload_one(api, manifest_src, dst, manifest_src.stat().st_size):
        log(f'FAILED {dst}')
        raise SystemExit(4)
    log('ALL_UPLOADED')


if __name__ == '__main__':
    main()
