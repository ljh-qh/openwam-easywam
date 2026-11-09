"""Generic 4-way parallel chunked uploader for ModelScope (resume-friendly)."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi

TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def remote_sizes(api, repo):
    out = {}
    for item in api.get_model_files(repo, recursive=True):
        if isinstance(item, dict):
            path = item.get('Path') or item.get('path')
            size = item.get('Size', item.get('size'))
        else:
            path = getattr(item, 'Path', None) or getattr(item, 'path', None)
            size = getattr(item, 'Size', None) or getattr(item, 'size', None)
        if path:
            out[str(path)] = int(size) if size is not None else None
    return out


def upload_one(api, repo, src, dst, size, retries=4):
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo, repo_type='model')
        except Exception as exc:  # noqa: BLE001
            log(f'RETRY {dst} attempt={attempt} {type(exc).__name__}: {str(exc)[:140]}')
            time.sleep(min(20 * attempt, 90))
            continue
        try:
            got = remote_sizes(api, repo).get(dst)
        except Exception as exc:  # noqa: BLE001
            log(f'VERIFY_RETRY {dst} {type(exc).__name__}: {str(exc)[:120]}')
            got = None
        elapsed = time.time() - started
        if got == size:
            log(f'DONE {dst} {size} {elapsed:.1f}s {size / elapsed / 1e6:.2f}MB/s')
            return True
        log(f'SIZE_MISMATCH {dst} remote={got} expected={size} attempt={attempt}')
        time.sleep(min(20 * attempt, 90))
    log(f'FAILED {dst}')
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--parts-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--parts-subdir', default='', help='subdir under prefix for chunk files')
    parser.add_argument('--extra', action='append', default=[], help='src:dst_rel')
    parser.add_argument('--extra-dir', action='append', default=[], help='src_dir:rel_prefix (recursive)')
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    manifest = json.loads((parts_dir / 'manifest.json').read_text())
    api = HubApi()
    api.login(TOKEN)
    existing = {}
    try:
        existing = remote_sizes(api, args.repo)
    except Exception as exc:  # noqa: BLE001
        log(f'REMOTE_LIST_FAILED {type(exc).__name__}: {str(exc)[:140]}')

    pending = []
    for part in manifest['parts']:
        key = f"{args.parts_subdir}/{part['name']}" if args.parts_subdir else part['name']
        dst = f"{args.prefix}/{key}"
        if existing.get(dst) == part['size']:
            log(f'SKIP {dst}')
            continue
        pending.append((parts_dir / part['name'], dst, part['size']))
    log(f'PENDING {len(pending)}/{len(manifest["parts"])} workers={args.workers}')

    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(upload_one, api, args.repo, src, dst, size): dst
                       for src, dst, size in pending}
            for future in as_completed(futures):
                dst = futures[future]
                try:
                    if not future.result():
                        failures.append(dst)
                except Exception as exc:  # noqa: BLE001
                    log(f'THREAD_FAILED {dst} {type(exc).__name__}: {str(exc)[:140]}')
                    failures.append(dst)
    if failures:
        log(f'FAILURES {failures}')
        raise SystemExit(2)

    extra_items = list(args.extra)
    for spec in args.extra_dir:
        src_dir_s, rel_prefix = spec.split(':', 1)
        base = Path(src_dir_s)
        for path in sorted(base.rglob('*')):
            if path.is_file():
                extra_items.append(f"{path}:{rel_prefix}/{path.relative_to(base)}")

    for spec in extra_items:
        src_s, dst_rel = spec.split(':', 1)
        src = Path(src_s)
        dst = f'{args.prefix}/{dst_rel}'
        size = src.stat().st_size
        if existing.get(dst) == size:
            log(f'SKIP {dst}')
            continue
        if not upload_one(api, args.repo, src, dst, size, retries=5):
            raise SystemExit(3)

    manifest_dst = f'{args.prefix}/upload_manifest.json'
    if not upload_one(api, args.repo, parts_dir / 'manifest.json', manifest_dst,
                      (parts_dir / 'manifest.json').stat().st_size, retries=5):
        raise SystemExit(4)
    log('ALL_UPLOADED')


if __name__ == '__main__':
    main()
