"""4-way parallel chunked upload for checkpoint_step_33050 (resume-friendly).

Each worker owns its HubApi instance and uploads one 1 GiB part with bounded
retries + remote size verification. Completed parts (remote size match or a
local DONE line) are skipped, never re-uploaded.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = 'ljhqh24/openwam'
PREFIX = 'checkpoints/top_state_5ep_33050'
PARTS = Path('/datapool/lianjh/openwam_top/work/parts_33050')
RUN = Path('/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56')
TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()
OLD_LOG = Path('/datapool/lianjh/openwam_top/logs/upload_33050_chunked.log')
WORKERS = 4


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def remote_sizes(api):
    out = {}
    for item in api.get_model_files(REPO, recursive=True):
        if isinstance(item, dict):
            path = item.get('Path') or item.get('path')
            size = item.get('Size', item.get('size'))
        else:
            path = getattr(item, 'Path', None) or getattr(item, 'path', None)
            size = getattr(item, 'Size', None) or getattr(item, 'size', None)
        if path:
            out[str(path)] = int(size) if size is not None else None
    return out


def upload_part(part, size):
    src = PARTS / part
    dst = f'{PREFIX}/parts/{part}'
    api = HubApi()
    api.login(TOKEN)
    for attempt in range(1, 5):
        started = time.time()
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
        except Exception as exc:  # noqa: BLE001
            log(f'RETRY {part} attempt={attempt} {type(exc).__name__}: {str(exc)[:140]}')
            time.sleep(min(20 * attempt, 90))
            continue
        try:
            got = remote_sizes(api).get(dst)
        except Exception as exc:  # noqa: BLE001
            log(f'VERIFY_RETRY {part} {type(exc).__name__}: {str(exc)[:120]}')
            got = None
        elapsed = time.time() - started
        if got == size:
            log(f'DONE {part} {size} {elapsed:.1f}s {size / elapsed / 1e6:.2f}MB/s')
            return part
        log(f'SIZE_MISMATCH {part} remote={got} expected={size} attempt={attempt}')
        time.sleep(min(20 * attempt, 90))
    log(f'FAILED {part}')
    return None


def main():
    manifest = json.loads((PARTS / 'manifest.json').read_text())
    api = HubApi()
    api.login(TOKEN)
    existing = {}
    try:
        existing = remote_sizes(api)
    except Exception as exc:  # noqa: BLE001
        log(f'REMOTE_LIST_FAILED {type(exc).__name__}: {str(exc)[:140]}')

    done_local = set()
    if OLD_LOG.exists():
        for line in OLD_LOG.read_text(errors='replace').splitlines():
            if 'DONE ' in line and '/parts/' in line:
                done_local.add(line.rsplit('/parts/', 1)[1].split()[0].strip())

    pending = []
    for part in manifest['parts']:
        name = part['name']
        dst = f"{PREFIX}/parts/{name}"
        if name in done_local or existing.get(dst) == part['size']:
            log(f'SKIP {name}')
            continue
        pending.append((name, part['size']))
    log(f'PENDING {len(pending)}/{len(manifest["parts"])} workers={WORKERS}')
    if not pending:
        log('PARTS_ALREADY_DONE')

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(upload_part, name, size): name for name, size in pending}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                log(f'THREAD_FAILED {name} {type(exc).__name__}: {str(exc)[:140]}')
                result = None
            if result is None:
                failures.append(name)
    if failures:
        log(f'FAILURES {failures}')
        raise SystemExit(2)

    extras = [(RUN / 'config.yaml', f'{PREFIX}/config.yaml'),
              (RUN / 'normalization_stats.npy', f'{PREFIX}/normalization_stats.npy')]
    for path in sorted((RUN / 'tokenizer').rglob('*')):
        if path.is_file():
            extras.append((path, f'{PREFIX}/tokenizer/{path.relative_to(RUN / "tokenizer")}'))
    for src, dst in extras:
        size = src.stat().st_size
        if existing.get(dst) == size:
            log(f'SKIP {dst}')
            continue
        log(f'UPLOAD {dst} {size}')
        # small files: single attempt with retries
        ok = False
        for attempt in range(1, 6):
            try:
                api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                log(f'RETRY {dst} {attempt} {type(exc).__name__}: {str(exc)[:120]}')
                time.sleep(min(20 * attempt, 90))
        if not ok:
            log(f'FAILED {dst}')
            raise SystemExit(3)
        log(f'DONE {dst} {size}')

    manifest_src = PARTS / 'manifest.json'
    dst = f'{PREFIX}/upload_manifest.json'
    for attempt in range(1, 6):
        try:
            api.upload_file(path_or_fileobj=str(manifest_src), path_in_repo=dst, repo_id=REPO, repo_type='model')
            break
        except Exception as exc:  # noqa: BLE001
            log(f'RETRY {dst} {attempt} {type(exc).__name__}: {str(exc)[:120]}')
            time.sleep(min(20 * attempt, 90))
    log('ALL_UPLOADED')


if __name__ == '__main__':
    main()
