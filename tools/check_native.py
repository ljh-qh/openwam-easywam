import json
import pyarrow.parquet as pq
root = '/datapool/lianjh/openwam_top/data/lerobot_native_58'
rep = json.load(open(root + '/conversion_report.json'))
print('excluded names:', [e['episode'] for e in rep['excluded']])
for e in rep['excluded']:
    print(' ', e['episode'], '-', e['reason'])
info = json.load(open(root + '/meta/info.json'))
print('info:', info['total_episodes'], info['total_frames'], info['fps'], info['splits'])
eps = pq.read_table(root + '/meta/episodes/chunk-000/file-000.parquet')
print('episodes rows:', eps.num_rows)
print('episode_index range:', eps['episode_index'][0].as_py(), eps['episode_index'][-1].as_py())
print('tasks:', pq.read_table(root + '/meta/tasks.parquet').to_pandas())
t = pq.read_table(root + '/data/chunk-000/file-000.parquet')
print('cols:', t.column_names)
print('rows0:', t.num_rows, 'state dim', len(t['observation.state'][0].as_py()))
d = t.to_pandas()
print('valid windows in ep0:', int(d['annotation.valid_window_start_33'].sum()))
print('action == next state:', bool((d['action'].iloc[:-1].tolist() == d['observation.state'].iloc[1:].tolist())))
