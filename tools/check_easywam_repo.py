from pathlib import Path
from modelscope.hub.api import HubApi

api = HubApi()
api.login(Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip())
files = api.get_model_files('ljhqh24/easywam', recursive=True)
rows = []
for f in files:
    p = str(f.get('Path', ''))
    if p.startswith('top_5ep/'):
        rows.append((p, f.get('Size')))
for p, s in sorted(rows):
    print(p, s)
print('TOTAL_FILES', len(rows))
