"""Convert audited, synchronized badminton LeRobot v3 joints to EEF80.

The source stays read-only. RGB videos are linked without another encode.
Run with the repository's .runtime/venv/bin/python; no GPU is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinematics import ACTIVE, ROTATION_DIMS, URDFKinematics, canonical80


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def replace(table, name, values, dtype):
    arr = pa.array(values.tolist(), type=dtype)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, arr)
    return table.append_column(name, arr)


def compute_statistics(action, state):
    """Fit action/state statistics over all valid demonstration rows."""
    stats = {}
    for name, values in [('action', action), ('observation.state', state)]:
        x = values.astype(np.float64)
        stats[name] = {key: value.tolist() for key, value in [('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)), ('std', x.std(0)), ('count', np.array([len(x)]))]}
    pooled = np.concatenate([action, state]).astype(np.float64)
    norm = {key: np.asarray(value, dtype=np.float32) for key, value in [('min', pooled.min(0)), ('max', pooled.max(0)), ('mean', pooled.mean(0)), ('std', pooled.std(0)), ('q01', np.quantile(pooled, .01, axis=0)), ('q99', np.quantile(pooled, .99, axis=0))]}
    for key, value in [('min', -1), ('max', 1), ('mean', 0), ('std', 1), ('q01', -1), ('q99', 1)]:
        norm[key][ROTATION_DIMS] = value
        norm[key][~ACTIVE] = value
    return stats, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing dataset: {target}')
    robot = URDFKinematics(args.urdf)
    info = json.loads((source / 'meta/info.json').read_text())
    if info['features']['action']['shape'] != [58]:
        raise ValueError('This converter requires the audited 58-D arm + Sharpa feedback source')
    # All demonstrations train the real-robot policy, regardless of source splits.
    info['splits'] = {'train': f"0:{info['total_episodes']}"}
    target.mkdir(parents=True)
    (target / 'provenance').mkdir()
    shutil.copy2(args.urdf, target / 'provenance/marvin_dual_arm.urdf')
    hashes = {}
    for p in sorted(source.glob('meta/**/*')):
        if p.is_file():
            hashes[str(p.relative_to(source))] = sha256(p)
    episodes = pq.read_table(source / 'meta/episodes/chunk-000/file-000.parquet').to_pandas()
    action_names = info['features']['action']['names']
    hands = [name.removeprefix('left_hand_joint_') for name in action_names[14:36]]
    sample_names = [f'reserved_{i}' for i in range(80)]
    for side, pos, hand in [('left', 0, 10), ('right', 34, 44)]:
        sample_names[pos:pos+9] = [f'{side}_tcp_{x}' for x in ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21')]
        sample_names[hand:hand+22] = [f'{side}_hand_joint_{name}' for name in hands]
    action_train, state_train = [], []
    records = []
    windows = {'train': 0}
    invalid_rows = 0
    for _, row in episodes.iterrows():
        ep = int(row.episode_index)
        rel = info['data_path'].format(chunk_index=int(row['data/chunk_index']), file_index=int(row['data/file_index']))
        if any(r['data_path'] == rel for r in records):
            raise ValueError('Expected one parquet per episode in the audited source')
        src = source / rel
        hashes[rel] = sha256(src)
        table = pq.read_table(src)
        n = len(table)
        if n != int(row.length) or not np.all(table['episode_index'].to_numpy() == ep):
            raise ValueError(f'Episode layout mismatch: {src}')
        alignment = json.loads((args.alignment / 'episodes' / f'episode_{ep+1:04d}.json').read_text())
        raw = Path(alignment['source_directory']) / 'robot_states.h5'
        if sha256(raw) != alignment['source_robot_sha256']:
            raise ValueError(f'Raw robot state changed since alignment: {raw}')
        joints = np.stack(table['observation.state'].to_numpy())
        next_joints = np.stack(table['action'].to_numpy())
        np.testing.assert_array_equal(next_joints[:-1], joints[1:])
        np.testing.assert_array_equal(next_joints[-1], joints[-1])
        with h5py.File(raw, 'r') as f:
            arm = f['robot_state/pico_teleop']
            arm_index = table['source.robot_row_index'].to_numpy()
            measured = np.deg2rad(np.concatenate([arm['real_left_deg'][:], arm['real_right_deg'][:]], axis=1)).astype(np.float32)
            np.testing.assert_allclose(joints[:, :14], measured[arm_index], atol=3e-7, rtol=0)
            hand = f['robot_state/sharpa_joints']
            if hand.attrs.get('source') != 'measured_feedback' or hand.attrs.get('units') != 'rad':
                raise ValueError(f'Unverified hand semantics: {raw}')
            if list(hand.attrs['joint_names']) != hands:
                raise ValueError('Hand joint order differs from metadata')
            hi = table['source.hand_row_index'].to_numpy()
            if np.any(hi < 0):
                raise ValueError('Missing hand feedback must be masked before conversion')
            np.testing.assert_array_equal(joints[:, 14:36], hand['left_rad'][:][hi])
            np.testing.assert_array_equal(joints[:, 36:58], hand['right_rad'][:][hi])
        state = canonical80(joints, robot)
        action = np.concatenate([state[1:], state[-1:]])
        time_valid = table['annotation.time_valid'].to_numpy().astype(bool)
        action_valid = time_valid & np.r_[time_valid[1:], False]
        np.testing.assert_array_equal(table['annotation.action_valid'].to_numpy(), action_valid)
        starts = np.zeros(n, dtype=bool)
        for start in range(max(0, n-32)):
            starts[start] = bool(time_valid[start:start+33].all() and action_valid[start:start+32].all())
        np.testing.assert_array_equal(starts, table['annotation.valid_window_start_33'].to_numpy())
        mask = np.broadcast_to(ACTIVE, (n, 80)).copy() & action_valid[:, None]
        table = replace(table, 'source.joint_state', joints, pa.list_(pa.float32(), 58))
        table = replace(table, 'source.joint_action', next_joints, pa.list_(pa.float32(), 58))
        table = replace(table, 'observation.state', state, pa.list_(pa.float32(), 80))
        table = replace(table, 'action', action, pa.list_(pa.float32(), 80))
        table = replace(table, 'action_mask', mask, pa.list_(pa.bool_(), 80))
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest, compression='zstd')
        windows['train'] += int(starts.sum())
        invalid_rows += int((~time_valid).sum())
        action_train.append(action[action_valid])
        state_train.append(state[time_valid])
        # Keep episode statistics consistent with rewritten features.
        for name, values, valid in [('action', action, action_valid), ('observation.state', state, time_valid)]:
            x = values[valid].astype(np.float64)
            if not len(x):
                raise ValueError(f'Episode {ep} has no valid {name}')
            for key, value in [('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)), ('std', x.std(0)), ('count', np.array([len(x)]))]:
                episodes.at[ep, f'stats/{name}/{key}'] = value
        records.append({'episode_index': ep, 'source_name': alignment['source_name'], 'data_path': rel, 'rows': n,
                        'valid_windows_33': int(starts.sum()), 'raw_robot_path': str(raw), 'raw_robot_sha256': alignment['source_robot_sha256']})
        print(f'Episode {ep+1}/{len(episodes)}: {n} rows, {starts.sum()} valid windows', flush=True)
    # All demonstrations, with a common physical scale for action and state.
    a, s = np.concatenate(action_train), np.concatenate(state_train)
    stats, norm = compute_statistics(a, s)
    inactive = np.flatnonzero(~ACTIVE)
    (target / 'meta/episodes/chunk-000').mkdir(parents=True)
    for column in episodes.columns:
        if column.startswith('stats/'):
            episodes[column] = episodes[column].map(lambda value: np.atleast_1d(value).tolist())
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False), target / 'meta/episodes/chunk-000/file-000.parquet')
    shutil.copy2(source / 'meta/tasks.parquet', target / 'meta/tasks.parquet')
    write_json(target / 'meta/stats.json', stats)
    np.save(target / 'meta/normalization_stats.npy', {'eef': norm})
    info['robot_type'] = 'marvin_dual_arm_sharpa_eef80'
    for feature in ('action', 'observation.state', 'action_mask'):
        info['features'][feature]['shape'] = [80]
        info['features'][feature]['names'] = sample_names
    for name in ('source.joint_state', 'source.joint_action'):
        info['features'][name] = {'dtype': 'float32', 'shape': [58], 'names': action_names}
    write_json(target / 'meta/info.json', info)
    video_records = []
    for path in sorted(source.glob('videos/**/*.mp4')):
        rel = path.relative_to(source)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(path.resolve())
        video_records.append({'path': str(rel), 'source': str(path), 'sha256': sha256(path)})
    contract = {
        'schema': 'openwam_canonical_eef80', 'action_source': 'next_measured_state', 'action_shift_frames': 1,
        'reference_frame': 'dual_arm_mount', 'position_units': 'meters', 'rotation': 'matrix first column then second column',
        'tcp_links': ['TCP_Link_L', 'TCP_Link_R'], 'arm_joint_order': [f'Joint{i}_L' for i in range(1, 8)] + [f'Joint{i}_R' for i in range(1, 8)],
        'joint_mapping_basis': 'real_left_deg/right_deg array order 1..7 mapped to URDF Joint1..7 without sign or offset changes; no external hardware calibration supplied',
        'hand_units': 'rad', 'hand_joint_names': hands, 'active_indices': np.flatnonzero(ACTIVE).tolist(),
        'masked_slots': inactive.tolist(), 'terminal_policy': 'repeat final state for storage, mask terminal action',
        'camera_layout': ['observation.images.head', 'observation.images.left_wrist', 'observation.images.right_wrist'],
        'camera_sources': {'head': 'main', 'left_wrist': 'wrist2', 'right_wrist': 'wrist'},
        'valid_window_policy': '33 consecutive time_valid rows; action[t:t+32] predicts state[t+1:t+33]; no crossing gaps or episodes',
        'normalization_population': f"all {len(records)} episodes; pooled valid action/state; rotation identity; no held-out split",
        'evaluation_protocol': 'real_robot_rollouts',
        'task_instruction': 'badminton', 'task_instruction_detail': 'source label retained, no invented instruction',
        'urdf_sha256': sha256(args.urdf), 'source': str(source), 'source_alignment': str(args.alignment.resolve()),
        'video_storage': 'absolute symlinks to existing synchronized videos; keep source dataset available',
    }
    write_json(target / 'meta/conversion.json', contract)
    write_json(target / 'provenance/source_hashes.json', hashes)
    write_json(target / 'provenance/episodes.json', records)
    write_json(target / 'provenance/videos.json', video_records)
    for rel, digest in hashes.items():
        if sha256(source / rel) != digest:
            raise ValueError(f'Source changed during conversion: {rel}')
    report = {'complete': True, 'episodes': len(records), 'rows': sum(r['rows'] for r in records), 'videos': len(video_records),
              'action_dim': 80, 'state_dim': 80, 'supervised_dims': int(ACTIVE.sum()), 'valid_windows_33': windows,
              'train_episodes': len(records), 'val_episodes': 0, 'evaluation_protocol': 'real_robot_rollouts', 'invalid_time_rows_excluded': invalid_rows,
              'train_action_rows': len(a), 'train_state_rows': len(s), 'source_numeric_and_hand_feedback_verified': True,
              'original_source_files_unchanged': True, 'urdf': str(args.urdf.resolve()), 'target': str(target),
              'hardware_calibration_verified': False}
    write_json(target / 'conversion_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
