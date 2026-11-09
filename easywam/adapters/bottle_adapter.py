"""Compact-62 EasyWAM view of the audited OpenWAM bottle samples.

Same contract as `badminton_adapter.py` (state[t+1], 62 ACTIVE dims, 9x384x320
RGB), but bound to the bottle dataset (197 episodes / 105748 windows) and
its instruction prompt. The window count and prompt are validated at startup
instead of being hard-coded to the 146-episode badminton release.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from openwam_badminton import BadmintonDataset
from kinematics import ACTIVE

EXPECTED_WINDOWS = 210041
EXPECTED_ACTIVE = 62


def _dataset_prompts(tasks_path: Path):
    """Return the set of task strings stored in a LeRobot v3 tasks.parquet."""
    table = pq.read_table(tasks_path)
    names = table.column_names
    metadata = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
    index_columns = [c for c in metadata.get("index_columns", []) if isinstance(c, str)]
    if "task" in names and "task" not in index_columns:
        column = table["task"]
    elif index_columns:
        column = table[index_columns[0]]
    elif "__index_level_0__" in names:
        column = table["__index_level_0__"]
    else:
        raise ValueError(f"Missing task text column in {tasks_path}")
    return {str(value) for value in column.to_pylist()}


class BottleEasyWAM(torch.utils.data.Dataset):
    def __init__(self, dataset_dirs, context_file, **unused):
        if len(dataset_dirs) != 1:
            raise ValueError('Exactly one audited dataset is required')
        self.dataset_dir = Path(dataset_dirs[0])
        self.context = torch.load(Path(context_file), map_location='cpu', weights_only=True)

        prompts = _dataset_prompts(self.dataset_dir / 'meta/tasks.parquet')
        if prompts != {self.context['prompt']}:
            raise ValueError(
                f"Dataset prompts {sorted(prompts)!r} do not match context prompt "
                f"{self.context['prompt']!r}")

        self.reader = BadmintonDataset(
            dataset_dirs[0], num_frames=33, video_stride=4, window_stride=1,
            height=384, width=320, split='train', multiview=True,
            camera_layout=['observation.images.head', 'observation.images.left_wrist',
                           'observation.images.right_wrist'],
            normalize_mode='min-max', unify_action=False, color_jitter=None,
        )
        self.active = torch.from_numpy(np.flatnonzero(ACTIVE))
        if len(self.active) != EXPECTED_ACTIVE:
            raise ValueError(f'Audited action layout changed: {len(self.active)} active dims')
        if len(self.reader) != EXPECTED_WINDOWS:
            raise ValueError(
                f'Expected {EXPECTED_WINDOWS} windows for bottle, got {len(self.reader)}')

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
