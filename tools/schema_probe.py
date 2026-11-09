import json
import pyarrow.parquet as pq
root = '/datapool/lianjh/openwam/data/openwam_eef80'
info = json.load(open(root + '/meta/info.json'))
print(json.dumps({k: v for k, v in info.items() if k != 'features'}, indent=1))
print('FEATURES:')
for k, v in info['features'].items():
    print(' ', k, v.get('dtype'), v.get('shape'), (v.get('names') or [])[:2])
t = pq.read_table(root + '/meta/episodes/chunk-000/file-000.parquet')
print('EPISODES SCHEMA:')
print(t.schema)
print('EPISODES ROW0:')
row = t.slice(0, 1).to_pylist()[0]
for k, v in row.items():
    s = str(v)
    print(' ', k, '=', s[:100])
print('TASKS:')
print(pq.read_table(root + '/meta/tasks.parquet').to_pandas())
