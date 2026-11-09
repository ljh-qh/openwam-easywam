"""Badminton reader variant: hybrid supervision + paraphrase-augmented prompts.

Arms are supervised from `source.arm_joint_command` (the teleop command) through
the same URDF FK used for the canonical EF80 layout; hands stay on the next
measured joint state because no hand command was ever recorded.
"""
import json
import random
from pathlib import Path

import numpy as np

from kinematics import ACTIVE, URDFKinematics, canonical80
from openwam.dataloader.registry import register_dataset
from openwam.dataloader.utils.normalization import apply_normalization
from openwam_badminton import BadmintonDataset


@register_dataset('badminton_hybrid')
class BadmintonHybrid(BadmintonDataset):
    NEEDED_COLS = tuple(BadmintonDataset.NEEDED_COLS) + (
        'source.joint_state', 'source.arm_joint_command')

    def __init__(self, dataset_dir, **kwargs):
        self._hybrid_stats_path = kwargs.pop('hybrid_stats', None)
        paraphrases_path = kwargs.pop('paraphrases', None)
        super().__init__(dataset_dir, **kwargs)
        self._hybrid_stats = None
        self._robot = None
        if self._hybrid_stats_path:
            stats = np.load(self._hybrid_stats_path, allow_pickle=True).item()['eef']
            expected = ('min', 'max', 'mean', 'std', 'q01', 'q99')
            if any(k not in stats or np.shape(stats[k]) != (80,) for k in expected):
                raise ValueError('hybrid stats must carry the six 80-dim vectors')
            self._hybrid_stats = stats
            self._normalization_stats = stats
            self._robot = URDFKinematics(Path(dataset_dir) / 'provenance/marvin_dual_arm.urdf')
        self._paraphrases = None
        if paraphrases_path:
            payload = json.loads(Path(paraphrases_path).read_text())
            self._paraphrases = list(payload['seen'])
            if not self._paraphrases:
                raise ValueError('paraphrase list is empty')

    def _action_20d(self, win):
        if self._hybrid_stats is None:
            return super()._action_20d(win)
        js = np.stack(win['source.joint_state'].to_numpy()).astype(np.float32)
        ac = np.stack(win['source.arm_joint_command'].to_numpy()).astype(np.float32)
        joints = np.concatenate([ac[1:], js[1:, 14:58]], axis=1)
        raw = canonical80(joints, self._robot)
        out = apply_normalization(raw, self._hybrid_stats, self._normalize_mode)
        out[:, ~ACTIVE] = 0
        return out

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self._paraphrases:
            sample['prompt'] = random.choice(self._paraphrases)
        return sample
