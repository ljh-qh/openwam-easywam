import os, time
from pathlib import Path
from modelscope.hub.api import HubApi

REPO = 'ljhqh24/openwam'
PREFIX = 'checkpoints/top_state_5ep_33050'
RUN = Path('/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56')
TOKEN = Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip()
files = [
    (RUN / 'checkpoint_step_33050.safetensors', f'{PREFIX}/checkpoint_step_33050.safetensors'),
    (RUN / 'config.yaml', f'{PREFIX}/config.yaml'),
    (RUN / 'normalization_stats.npy', f'{PREFIX}/normalization_stats.npy'),
]
for p in sorted((RUN / 'tokenizer').rglob('*')):
    if p.is_file():
        files.append((p, f'{PREFIX}/tokenizer/{p.relative_to(RUN / "tokenizer")}'))

api = HubApi()
api.login(TOKEN)
try:
    m = api.get_model(REPO)
    print('REPO_OK', getattr(m, 'ModelId', REPO), flush=True)
except Exception as e:
    print('REPO_CHECK_FAILED', type(e).__name__, str(e)[:200], flush=True)

for src, dst in files:
    size = src.stat().st_size
    print(f'UPLOAD {dst} {size} bytes', flush=True)
    for attempt in range(1, 6):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=REPO, repo_type='model')
            print(f'DONE {dst}', flush=True)
            break
        except Exception as e:
            print(f'RETRY {attempt} {type(e).__name__} {str(e)[:200]}', flush=True)
            if attempt == 5:
                raise
            time.sleep(60)
print('ALL_UPLOADED', flush=True)
