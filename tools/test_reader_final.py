import sys
sys.path.insert(0, '/datapool/lianjh/openwam/code/source')
sys.path.insert(0, '/datapool/lianjh/openwam/code/badminton')
from openwam_badminton import BadmintonDataset
root = '/datapool/lianjh/openwam_top/data/openwam_eef80'
ds = BadmintonDataset(root, num_frames=33, video_stride=4, window_stride=1, height=384, width=320,
                      multiview=True,
                      camera_layout=['observation.images.head', 'observation.images.left_wrist', 'observation.images.right_wrist'],
                      normalize_mode='min-max', split='train')
print('LEN', len(ds))
s = ds[len(ds) - 1]
print('last video', len(s['video']), 'action', tuple(s['action'].shape), 'proprio', tuple(s['proprio'].shape))
print('prompt', s['prompt'])
print('normalization stats', {k: float(v[0]) for k, v in list(ds.normalization_stats.items())[:2]})
print('FINAL_READER_OK')
