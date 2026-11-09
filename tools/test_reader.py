import sys
sys.path.insert(0, '/datapool/lianjh/openwam/code/source')
sys.path.insert(0, '/datapool/lianjh/openwam/code/badminton')
from openwam_badminton import BadmintonDataset
root = '/datapool/lianjh/openwam_top/data/smoke_eef80'
ds = BadmintonDataset(root, num_frames=33, video_stride=4, window_stride=1, height=384, width=320,
                      multiview=True,
                      camera_layout=['observation.images.head', 'observation.images.left_wrist', 'observation.images.right_wrist'],
                      normalize_mode='min-max', split='train')
print('LEN', len(ds))
s = ds[0]
print('video_frames', len(s['video']), 'size', getattr(s['video'][0], 'size', None))
print('action', tuple(s['action'].shape), 'proprio', tuple(s['proprio'].shape), 'mask', tuple(s['action_mask'].shape))
print('prompt', s['prompt'])
print('first_frame', len(s['first_frame_image']))
print('READER_OK')
