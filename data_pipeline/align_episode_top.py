#!/usr/bin/env python3
"""Align episode_top raw recordings into a 58-D LeRobot v3 dataset.

30 Hz nearest-neighbour grid (max 50 ms), measured arm joints (pico_teleop
real_left_deg/real_right_deg, deg -> rad) and measured Sharpa hand feedback.
Videos are fully decoded and re-encoded so video frame k == grid row k.

Outputs:
  <native>/data/chunk-000/file-XXX.parquet   one parquet per episode
  <native>/videos/<camera>/...               re-encoded 30 fps video
  <native>/meta/{info.json,stats.json,tasks.parquet,episodes/...}
  <aligned>/episodes/episode_XXXX.{json,npz}  alignment audit for convert_to_openwam.py
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ('main', 'wrist', 'wrist2')
VIDEO_KEYS = {
    'main': 'observation.images.head',
    'wrist2': 'observation.images.left_wrist',
    'wrist': 'observation.images.right_wrist',
}
FPS = 30
WINDOW = 33
MAX_ERROR_NS = 50_000_000
INSTRUCTION = 'pull the string to spin the top on the table'
TEMPLATE_INFO = Path('/datapool/lianjh/openwam/data/openwam_eef80/meta/info.json')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def nearest(times, grid):
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right)


def align_episode(raw: Path, hand_names):
    manifest = json.loads((raw / 'manifest.json').read_text())
    if manifest.get('status') != 'closed':
        raise ValueError(f"manifest status={manifest.get('status')}")
    if manifest.get('recording_errors'):
        raise ValueError(f"recording_errors={manifest['recording_errors']}")
    with h5py.File(raw / 'robot_states.h5', 'r') as f:
        if 'robot_state/pico_teleop' not in f:
            raise ValueError('missing robot_state/pico_teleop')
        group = f['robot_state/pico_teleop']
        for key in ('sys_time_ns', 'real_left_deg', 'real_right_deg', 'command_rad'):
            if key not in group:
                raise ValueError(f'missing pico_teleop/{key}')
        robot = {key: group[key][:] for key in (
            'sys_time_ns', 'real_left_deg', 'real_right_deg', 'command_rad',
            'sim_tcp_left_pos', 'sim_tcp_left_quat', 'sim_tcp_right_pos', 'sim_tcp_right_quat') if key in group}
        if 'robot_state/sharpa_joints' not in f:
            raise ValueError('missing robot_state/sharpa_joints')
        hand = f['robot_state/sharpa_joints']
        if hand.attrs.get('units') != 'rad' or hand.attrs.get('source') != 'measured_feedback':
            raise ValueError('unverified hand feedback semantics')
        if list(hand.attrs['joint_names']) != hand_names:
            raise ValueError('hand joint order mismatch')
        hands = {key: hand[key][:] for key in ('sys_time_ns', 'left_rad', 'right_rad')}

    camera_times = {}
    for camera in CAMERAS:
        entries = [json.loads(line) for line in
                   (raw / camera / 'timestamps_take001.jsonl').read_text().splitlines() if line.strip()]
        frames = [row for row in entries if 'i' in row]
        if [int(row['i']) for row in frames] != list(range(1, len(frames) + 1)):
            raise ValueError(f'non-contiguous camera frame indices: {raw}/{camera}')
        camera_times[camera] = np.rint(np.array([row['mono'] for row in frames]) * 1e9).astype(np.int64)
        if manifest['cameras'][camera]['frames'] != len(frames):
            raise ValueError(f'camera manifest/timestamps disagree: {raw}/{camera}')

    robot_t = robot['sys_time_ns'].astype(np.int64)
    hand_t = hands['sys_time_ns'].astype(np.int64)
    streams = [robot_t] + [camera_times[c] for c in CAMERAS] + [hand_t]
    for times in streams:
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f'timestamps must be strictly increasing: {raw}')

    first = max(int(times[0]) for times in streams)
    last = min(int(times[-1]) for times in streams)
    n = ((last - first) * FPS) // 1_000_000_000 + 1
    if n < WINDOW:
        raise ValueError(f'insufficient common overlap: {n} rows')
    grid = first + np.rint(np.arange(n) * 1e9 / FPS).astype(np.int64)
    indices = [nearest(times, grid) for times in streams]
    errors = np.stack([times[index] - grid for times, index in zip(streams, indices)], axis=1)
    valid = (np.abs(errors) <= MAX_ERROR_NS).all(axis=1)

    ri, hi = indices[0], indices[-1]
    measured = np.deg2rad(np.concatenate(
        [robot['real_left_deg'], robot['real_right_deg']], axis=1)).astype(np.float32)
    joints = np.concatenate([measured[ri], hands['left_rad'][hi], hands['right_rad'][hi]], axis=1).astype(np.float32)
    if joints.shape != (n, 58) or not np.isfinite(joints).all():
        raise ValueError(f'missing/nonfinite measured joints: {joints.shape}')

    flags = np.zeros(n, dtype=bool)
    flags[:n - 32] = np.convolve(valid.astype(np.int64), np.ones(WINDOW, dtype=np.int64), mode='valid') == WINDOW
    if not flags.any():
        raise ValueError('no valid 33-frame training windows')

    def pose(pos_key, quat_key):
        if pos_key in robot and quat_key in robot:
            return np.concatenate([robot[pos_key][ri], robot[quat_key][ri]], axis=1).astype(np.float32)
        return np.zeros((n, 7), dtype=np.float32)

    return {
        'name': raw.name,
        'manifest_episode': manifest.get('episode'),
        'robot_sha256': sha256(raw / 'robot_states.h5'),
        'grid': grid,
        'robot_row_index': ri.astype(np.int64),
        'hand_row_index': hi.astype(np.int64),
        'camera_indices': np.stack(indices[1:4], axis=1).astype(np.int64),
        'errors': errors.astype(np.int64),
        'valid': valid,
        'flags': flags,
        'joints': joints,
        'command_rad': robot['command_rad'][ri].astype(np.float32),
        'sim_tcp_left': pose('sim_tcp_left_pos', 'sim_tcp_left_quat'),
        'sim_tcp_right': pose('sim_tcp_right_pos', 'sim_tcp_right_quat'),
    }


def encode_aligned(source, target, mapping, expected_frames):
    """Fully decode the raw video; encode only timestamp-selected RGB frames."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.partial.mp4')
    source_count = output_count = 0
    counts = np.bincount(mapping, minlength=expected_frames)
    with av.open(str(source)) as inp, av.open(str(temporary), 'w') as out:
        input_stream = inp.streams.video[0]
        input_stream.thread_count = 1
        stream = out.add_stream('libx264', rate=FPS)
        stream.width, stream.height = input_stream.width, input_stream.height
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.thread_count = 2
        stream.options = {'crf': '18', 'preset': 'fast', 'g': '30'}
        for index, frame in enumerate(inp.decode(input_stream)):
            source_count += 1
            if index >= len(counts) or counts[index] == 0:
                continue
            rgb = frame.to_ndarray(format='rgb24')
            for _ in range(int(counts[index])):
                encoded = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                encoded.pts, encoded.time_base = output_count, Fraction(1, FPS)
                for packet in stream.encode(encoded):
                    out.mux(packet)
                output_count += 1
        for packet in stream.encode():
            out.mux(packet)
    if source_count != expected_frames or output_count != len(mapping):
        raise ValueError(f'decoded frame count mismatch: {source_count}/{expected_frames} -> {output_count}/{len(mapping)}')
    with av.open(str(temporary)) as check:
        if check.streams.video[0].frames != output_count:
            raise ValueError(f'encoded frame count mismatch: {temporary}')
    temporary.replace(target)
    return {'source_decoded_frames': source_count, 'output_frames': output_count,
            'width': stream.width, 'height': stream.height}


def prepare_episode(task):
    ep, ep_number, data, offset, native, aligned, schema, ep_schema, instruction = task
    n = len(data['grid'])
    ep_row = {
        'episode_index': ep, 'length': n, 'tasks': [instruction],
        'dataset_from_index': offset, 'dataset_to_index': offset + n,
        'data/chunk_index': 0, 'data/file_index': ep,
        'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
        'valid_windows_33': int(data['flags'].sum()),
    }
    joints = data['joints']
    valid = data['valid']
    action_valid = valid & np.r_[valid[1:], False]
    action = np.concatenate([joints[1:], joints[-1:]])
    columns = {
        'action': action.tolist(),
        'observation.state': joints.tolist(),
        'timestamp': (np.arange(n) / FPS).astype(np.float64),
        'frame_index': np.arange(n, dtype=np.int64),
        'episode_index': np.full(n, ep, dtype=np.int64),
        'index': np.arange(offset, offset + n, dtype=np.int64),
        'task_index': np.zeros(n, dtype=np.int64),
        'annotation.time_valid': valid,
        'annotation.action_valid': action_valid,
        'action_mask': np.broadcast_to(action_valid[:, None], (n, 58)).tolist(),
        'annotation.valid_window_start_33': data['flags'],
        'source.monotonic_time_ns': data['grid'],
        'source.robot_row_index': data['robot_row_index'],
        'source.hand_row_index': data['hand_row_index'],
        'source.arm_joint_command': data['command_rad'].tolist(),
        'source.time_error_ns': data['errors'].tolist(),
        'observation.sim_tcp_left_pose': data['sim_tcp_left'].tolist(),
        'observation.sim_tcp_right_pose': data['sim_tcp_right'].tolist(),
        'source.main.frame_index': data['camera_indices'][:, 0],
        'source.wrist.frame_index': data['camera_indices'][:, 1],
        'source.wrist2.frame_index': data['camera_indices'][:, 2],
    }
    table = pa.Table.from_pydict(columns, schema=schema)
    rel = f'data/chunk-000/file-{ep:03d}.parquet'
    (native / rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, native / rel, compression='zstd')

    for feature, values, mask in (('action', action, action_valid), ('observation.state', joints, valid)):
        x = values[mask].astype(np.float64)
        for key, value in (('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)),
                           ('std', x.std(0)), ('count', np.array([len(x)]))):
            ep_row[f'stats/{feature}/{key}'] = value.tolist()

    raw = Path(data['raw_dir'])
    camera_report = {}
    for camera in CAMERAS:
        key = VIDEO_KEYS[camera]
        video_rel = f'videos/{key}/chunk-000/file-{ep:03d}.mp4'
        expected = int(data['camera_frames'][camera])
        camera_report[camera] = encode_aligned(raw / camera / 'color_take001.mp4',
                                               native / video_rel, data['camera_indices'][:, CAMERAS.index(camera)],
                                               expected)
        camera_report[camera]['source_sha256'] = sha256(raw / camera / 'color_take001.mp4')
        camera_report[camera]['source_path'] = str(raw / camera / 'color_take001.mp4')
        ep_row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': ep,
                       f'videos/{key}/from_timestamp': 0.0, f'videos/{key}/to_timestamp': n / FPS})

    npz_name = f'episode_{ep_number:04d}.npz'
    np.savez_compressed(
        aligned / 'episodes' / npz_name,
        joints=joints,
        aligned_monotonic_time_ns=data['grid'],
        robot_row_index=data['robot_row_index'],
        hand_row_index=data['hand_row_index'],
        camera_frame_indices=data['camera_indices'],
        time_error_ns=data['errors'],
        time_valid=valid,
        valid_window_starts_33=np.flatnonzero(data['flags']),
    )
    report = {
        'episode_index': ep, 'source_name': data['name'], 'source_directory': str(raw),
        'source_robot_sha256': data['robot_sha256'], 'archive': f'episodes/{npz_name}',
        'aligned_rows': n, 'valid_rows': int(valid.sum()), 'invalid_rows': int((~valid).sum()),
        'valid_windows_33': int(data['flags'].sum()),
        'maximum_alignment_error_ms': dict(zip(
            ['robot', *CAMERAS, 'hands'], (np.abs(data['errors']).max(0) / 1e6).tolist())),
        'source_manifest_episode': data['manifest_episode'], 'source_recording_errors': [],
        'cameras': camera_report,
    }
    write_json(aligned / 'episodes' / f'episode_{ep_number:04d}.json', report)
    return ep_row


def build_schema():
    f32 = pa.float32()
    return pa.schema([
        ('action', pa.list_(f32, 58)),
        ('observation.state', pa.list_(f32, 58)),
        ('timestamp', pa.float64()),
        ('frame_index', pa.int64()),
        ('episode_index', pa.int64()),
        ('index', pa.int64()),
        ('task_index', pa.int64()),
        ('annotation.time_valid', pa.bool_()),
        ('annotation.action_valid', pa.bool_()),
        ('action_mask', pa.list_(pa.bool_(), 58)),
        ('annotation.valid_window_start_33', pa.bool_()),
        ('source.monotonic_time_ns', pa.int64()),
        ('source.robot_row_index', pa.int64()),
        ('source.hand_row_index', pa.int64()),
        ('source.arm_joint_command', pa.list_(f32, 14)),
        ('source.time_error_ns', pa.list_(pa.int64())),
        ('observation.sim_tcp_left_pose', pa.list_(f32, 7)),
        ('observation.sim_tcp_right_pose', pa.list_(f32, 7)),
        ('source.main.frame_index', pa.int64()),
        ('source.wrist.frame_index', pa.int64()),
        ('source.wrist2.frame_index', pa.int64()),
    ])


def episode_schema():
    fields = [
        ('episode_index', pa.int64()), ('length', pa.int64()), ('tasks', pa.list_(pa.string())),
        ('dataset_from_index', pa.int64()), ('dataset_to_index', pa.int64()),
        ('data/chunk_index', pa.int64()), ('data/file_index', pa.int64()),
        ('meta/episodes/chunk_index', pa.int64()), ('meta/episodes/file_index', pa.int64()),
    ]
    for feature in ('action', 'observation.state'):
        for stat in ('min', 'max', 'mean', 'std'):
            fields.append((f'stats/{feature}/{stat}', pa.list_(pa.float64())))
        fields.append((f'stats/{feature}/count', pa.list_(pa.int64())))
    for key in VIDEO_KEYS.values():
        fields += [(f'videos/{key}/chunk_index', pa.int64()),
                   (f'videos/{key}/file_index', pa.int64()),
                   (f'videos/{key}/from_timestamp', pa.float64()),
                   (f'videos/{key}/to_timestamp', pa.float64())]
    fields.append(('valid_windows_33', pa.int64()))
    return pa.schema(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('/datapool/lianjh/openwam_top/raw/episode_top'))
    parser.add_argument('--native', type=Path, default=Path('/datapool/lianjh/openwam_top/data/lerobot_native_58'))
    parser.add_argument('--aligned', type=Path, default=Path('/datapool/lianjh/openwam_top/data/aligned_native'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--instruction', default=INSTRUCTION)
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    native, aligned = args.native.resolve(), args.aligned.resolve()
    if native.exists() or aligned.exists():
        raise FileExistsError(f'refusing to replace existing output: {native} / {aligned}')

    hand_names = None
    candidates, failures = [], []
    for raw in sorted(raw_root.glob('episode_*')):
        try:
            with h5py.File(raw / 'robot_states.h5', 'r') as f:
                names = list(f['robot_state/sharpa_joints'].attrs['joint_names'])
            if hand_names is None:
                hand_names = names
            elif names != hand_names:
                raise ValueError('hand joint order differs between episodes')
            data = align_episode(raw, hand_names)
            data['raw_dir'] = str(raw)
            candidates.append(data)
        except Exception as exc:  # noqa: BLE001 - report and skip bad recordings
            failures.append({'episode': raw.name, 'reason': f'{type(exc).__name__}: {exc}'})

    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        raise SystemExit('no usable episodes')

    # camera frame counts come from the manifest (already validated against timestamps)
    for data in candidates:
        raw = Path(data['raw_dir'])
        manifest = json.loads((raw / 'manifest.json').read_text())
        data['camera_frames'] = {c: int(manifest['cameras'][c]['frames']) for c in CAMERAS}

    native.mkdir(parents=True)
    (aligned / 'episodes').mkdir(parents=True)
    template = json.loads(TEMPLATE_INFO.read_text())
    template_features = template['features']
    schema = build_schema()
    ep_schema = episode_schema()

    lengths = [len(d['grid']) for d in candidates]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_episode, (ep, ep + 1, data, int(offsets[ep]), native, aligned,
                                                 schema, ep_schema, args.instruction)): ep
                   for ep, data in enumerate(candidates)}
        for future in as_completed(futures):
            ep = futures[future]
            row = future.result()
            rows.append(row)
            print(f'[{len(rows)}/{len(candidates)}] episode {ep} rows={row["length"]} '
                  f'windows={row["valid_windows_33"]}', flush=True)
    rows.sort(key=lambda row: row['episode_index'])

    total_frames = int(sum(lengths))
    info = {k: v for k, v in template.items() if k != 'features'}
    info.update(robot_type='marvin_dual_arm_sharpa_58', total_episodes=len(rows), total_frames=total_frames,
                total_tasks=1, fps=FPS, splits={'train': f'0:{len(rows)}'})
    # 58-D layout: 14 arm joints (left 1..7, right 1..7) then 44 hand joints.
    names58 = ([f'left_arm_joint_{i}' for i in range(1, 8)]
               + [f'right_arm_joint_{i}' for i in range(1, 8)]
               + [f'left_hand_joint_{name}' for name in hand_names]
               + [f'right_hand_joint_{name}' for name in hand_names])
    features = dict(template_features)
    for key in ('source.joint_state', 'source.joint_action'):
        features.pop(key, None)
    for key in ('action', 'observation.state', 'action_mask'):
        features[key] = dict(features[key], shape=[58], names=names58)
    info['features'] = features
    write_json(native / 'meta/info.json', info)

    episodes_table = pa.Table.from_pylist(rows, schema=ep_schema)
    (native / 'meta/episodes/chunk-000').mkdir(parents=True)
    pq.write_table(episodes_table, native / 'meta/episodes/chunk-000/file-000.parquet')

    import pandas as pd
    tasks = pd.DataFrame({'task_index': np.array([0], dtype=np.int64)},
                         index=pd.Index([args.instruction], name='task'))
    tasks.to_parquet(native / 'meta/tasks.parquet')

    agg = {'action': [], 'observation.state': []}
    for row in rows:
        t = pq.read_table(native / f'data/chunk-000/file-{row["episode_index"]:03d}.parquet')
        for feature, valid_col in (('action', 'annotation.action_valid'), ('observation.state', 'annotation.time_valid')):
            agg[feature].append(np.stack(t[feature].to_numpy())[t[valid_col].to_numpy()])
    stats = {}
    for feature, arrays in agg.items():
        x = np.concatenate(arrays).astype(np.float64)
        stats[feature] = {key: value.tolist() for key, value in (
            ('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)), ('std', x.std(0)),
            ('count', np.array([len(x)])))}
    write_json(native / 'meta/stats.json', stats)

    report = {
        'complete': True, 'total_episodes': len(rows), 'total_frames': total_frames,
        'valid_windows_33': int(sum(row['valid_windows_33'] for row in rows)),
        'instruction': args.instruction, 'fps': FPS, 'max_time_error_ms': MAX_ERROR_NS / 1e6,
        'raw_directory': str(raw_root), 'excluded': failures,
        'hand_joint_names': hand_names,
        'source_numeric_and_hand_feedback_verified': True,
    }
    write_json(aligned / 'audit_report.json', report)
    write_json(native / 'conversion_report.json', report)
    print(json.dumps(report, indent=2))
    print('ALIGN_DONE')


if __name__ == '__main__':
    main()
