"""Compact-62 EasyWAM view of the audited OpenWAM badminton samples."""
from pathlib import Path
import numpy as np
import torch
from openwam_badminton import BadmintonDataset
from kinematics import ACTIVE


class BadmintonEasyWAM(torch.utils.data.Dataset):
    def __init__(self, dataset_dirs, context_file, **unused):
        if len(dataset_dirs) != 1:
            raise ValueError('Exactly one audited dataset is required')
        self.reader = BadmintonDataset(
            dataset_dirs[0], num_frames=33, video_stride=4, window_stride=1,
            height=384, width=320, split='train', multiview=True,
            camera_layout=['observation.images.head', 'observation.images.left_wrist',
                           'observation.images.right_wrist'],
            normalize_mode='min-max', unify_action=False, color_jitter=None,
        )
        self.active = torch.from_numpy(np.flatnonzero(ACTIVE))
        if len(self.active) != 62 or len(self.reader) != 114243:
            raise ValueError('Audited action layout or window count changed')
        self.context = torch.load(Path(context_file), map_location='cpu', weights_only=True)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        s = self.reader[index]
        action = s['action'].index_select(-1, self.active)
        proprio = s['proprio'].index_select(-1, self.active)
        if not s['action_mask'].index_select(-1, self.active).all():
            raise ValueError('Invalid action supervision reached the training loader')
        if not s['proprio_mask'].index_select(-1, self.active).all() or not s['video_mask'].all():
            raise ValueError('Invalid observation reached the training loader')
        if s['prompt'] != self.context['prompt']:
            raise ValueError(f"Unexpected prompt: {s['prompt']!r}")
        video = torch.from_numpy(np.stack([np.asarray(im).copy() for im in s['video']]))
        video = video.permute(3, 0, 1, 2).float().div(127.5).sub(1)
        if video.shape != (3, 9, 384, 320) or action.shape != (32, 62):
            raise ValueError('Unexpected sample geometry')
        return dict(video=video, action=action, proprio=proprio,
                    context=self.context['context'], context_mask=self.context['context_mask'],
                    prompt=s['prompt'], action_is_pad=torch.zeros(32, dtype=torch.bool),
                    image_is_pad=torch.zeros(9, dtype=torch.bool),
                    proprio_is_pad=torch.zeros(1, dtype=torch.bool))
