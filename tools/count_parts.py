import json
from pathlib import Path
from modelscope.hub.api import HubApi
api = HubApi()
api.login(Path('/datapool/lianjh/easywam/official/.ms_token').read_text().strip())
files = api.get_model_files('ljhqh24/openwam', recursive=True)
prefix = 'checkpoints/top_state_5ep_33050/'
parts = {str(f.get('Path')): f.get('Size') for f in files if str(f.get('Path','')).startswith(prefix + 'parts/')}
ok = {k: v for k, v in parts.items() if v == 1073741824}
print('REMOTE_PARTS', len(parts), 'FULL_1G', len(ok))
print('MISSING', sorted(set(f'part_{i:02d}' for i in range(24)) - {k.rsplit('/',1)[1].split()[0] for k in parts}))
