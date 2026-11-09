"""OpenWAM reader for the converted canonical EEF80 badminton dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.registry import register_dataset
from openwam.dataloader.utils.normalization import apply_normalization

from kinematics import ACTIVE


@register_dataset('badminton')
class BadmintonDataset(LeRobotV3Reader):
    DATASET_NAME = 'BadmintonEEF80'
    ACTION_DIM = 80
    ACTION_DIM_MASK = ACTIVE
    HEAD_CAMERA = 'observation.images.head'
    LEFT_WRIST_CAMERA = 'observation.images.left_wrist'
    RIGHT_WRIST_CAMERA = 'observation.images.right_wrist'
    WRIST_DECODE_TOLERATED = ()
    NEEDED_COLS = ('task_index', 'action', 'observation.state', 'action_mask',
                   'annotation.time_valid', 'annotation.action_valid', 'annotation.valid_window_start_33')

    def __init__(self, dataset_dir, **kwargs):
        if kwargs.get('unify_action', False):
            raise ValueError('Dataset already uses canonical 80-D slots; set unify_action=false')
        if int(kwargs.get('num_frames', 33)) != 33:
            raise ValueError('This audit uses 33-row windows; num_frames must be 33')
        if int(kwargs.get('window_stride', 1)) != 1:
            raise ValueError('Quality-indexed windows require window_stride=1')
        root = Path(dataset_dir)
        if kwargs.get('split', 'train') != 'train':
            raise ValueError('All badminton episodes are for training; evaluation uses real-robot rollouts')
        report = json.loads((root / 'conversion_report.json').read_text())
        if not report.get('complete') or report['action_dim'] != 80:
            raise ValueError('Dataset conversion has not completed')
        super().__init__(dataset_dir, **kwargs)
        if not len(self):
            raise ValueError(f'No valid windows for split {self._split}')

    def _train_min_window_len(self):
        return 33

    def _filter_episodes(self, episodes):
        # Represent each contiguous run of valid STARTS as a virtual range.
        # Preserve file-local offsets: never concatenate rows across a gap.
        ranges = []
        for _, row in episodes.iterrows():
            table = self._load_data_table(int(row['data/chunk_index']), int(row['data/file_index']))
            start = int(row['_data_row_offset'])
            win = table.slice(start, int(row.length))
            flags = win['annotation.valid_window_start_33'].to_numpy().astype(bool)
            times = win['annotation.time_valid'].to_numpy().astype(bool)
            expected = np.zeros(len(win), dtype=bool)
            if len(win) >= 33:
                expected[:len(win)-32] = np.convolve(times.astype(np.int64), np.ones(33, dtype=np.int64), mode='valid') == 33
            np.testing.assert_array_equal(flags, expected)
            masks = np.stack(win['action_mask'].to_numpy())
            valid = win['annotation.action_valid'].to_numpy().astype(bool)
            np.testing.assert_array_equal(masks, valid[:, None] & ACTIVE[None, :])
            edges = np.diff(np.r_[False, flags, False].astype(np.int8))
            for first, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                segment = row.copy()
                segment['_valid_start'] = int(first)
                segment['_valid_end'] = int(end + 32)
                ranges.append(segment)
        if not ranges:
            return episodes.iloc[:0].assign(_valid_start=0, _valid_end=0)
        return pd.DataFrame(ranges).reset_index(drop=True)

    def _load_stats(self, info):
        if self._normalize_mode not in (None, 'none', 'null', 'min-max', 'z-score', 'quantile'):
            raise ValueError(f'Unsupported normalization: {self._normalize_mode}')
        path = self._dataset_dir / 'meta/normalization_stats.npy'
        stats = np.load(path, allow_pickle=True).item()['eef']
        for key, value in stats.items():
            if np.shape(value) != (80,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key} normalization statistics')
        self.normalization_stats_path = str(path)
        return stats

    def _action_20d(self, win):
        # action[t] already stores state[t+1]; do not shift it a second time.
        raw = np.stack(win['action'].to_numpy()).astype(np.float32)
        result = apply_normalization(raw, self._normalization_stats, self._normalize_mode)
        result[:, ~ACTIVE] = 0
        return result

    def _proprio_20d(self, win):
        raw = np.stack(win['observation.state'].iloc[:1].to_numpy()).astype(np.float32)
        result = apply_normalization(raw, self._normalization_stats, self._normalize_mode)
        result[:, ~ACTIVE] = 0
        return result

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._getitem_impl(index)
