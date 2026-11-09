"""Align newly downloaded recordings and extend the audited native LeRobot v3.

Original recordings and the previous 91-episode dataset remain unchanged.
All large outputs are written below --root on /aifs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_to_openwam import sha256, write_json

CAMERAS = ('main', 'wrist', 'wrist2')
VIDEO_KEYS = {'main': 'observation.images.head', 'wrist2': 'observation.images.left_wrist', 'wrist': 'observation.images.right_wrist'}


def nearest(times, grid):
    right = np.clip(np.searchsorted(times, grid), 0, len(times)-1)
    left = np.maximum(0, right-1)
    return np.where(np.abs(times[left]-grid) <= np.abs(times[right]-grid), left, right)


def align_episode(raw, hand_names):
    manifest = json.loads((raw / 'manifest.json').read_text())
    if manifest.get('status') != 'closed' or manifest.get('recording_errors'):
        raise ValueError(f'Incomplete recording: {raw}')
    with h5py.File(raw / 'robot_states.h5', 'r') as f:
        group = f['robot_state/pico_teleop']
        robot = {name: value[:] for name, value in group.items()}
        hand = f['robot_state/sharpa_joints']
        if hand.attrs.get('units') != 'rad' or hand.attrs.get('source') != 'measured_feedback':
            raise ValueError(f'Unverified hand feedback: {raw}')
        if list(hand.attrs['joint_names']) != hand_names:
            raise ValueError(f'Hand joint order changed: {raw}')
        hands = {name: value.asstr()[:] if value.dtype.kind == 'O' else value[:] for name, value in hand.items()}
    camera_times = {}
    for camera in CAMERAS:
        entries = [json.loads(line) for line in (raw / camera / 'timestamps_take001.jsonl').read_text().splitlines() if line.strip()]
        frames = [row for row in entries if 'i' in row]
        if [int(row['i']) for row in frames] != list(range(1, len(frames)+1)):
            raise ValueError(f'Non-contiguous camera frame indices: {raw}/{camera}')
        camera_times[camera] = np.rint(np.array([row['mono'] for row in frames]) * 1e9).astype(np.int64)
        if manifest['cameras'][camera]['frames'] != len(frames):
            raise ValueError(f'Camera manifest/timestamps disagree: {raw}/{camera}')
    streams = [robot['sys_time_ns']] + [camera_times[camera] for camera in CAMERAS] + [hands['sys_time_ns']]
    for times in streams:
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f'Timestamps must be strictly increasing: {raw}')
    first = max(int(times[0]) for times in streams)
    last = min(int(times[-1]) for times in streams)
    n = ((last-first)*30)//1_000_000_000 + 1
    if n < 33:
        raise ValueError(f'Insufficient common overlap: {raw}')
    grid = first + np.rint(np.arange(n)*1e9/30).astype(np.int64)
    indices = [nearest(times, grid) for times in streams]
    errors = np.stack([times[index]-grid for times,index in zip(streams, indices)], axis=1)
    valid = (np.abs(errors) <= 50_000_000).all(axis=1)
    ri, hi = indices[0], indices[-1]
    measured = np.deg2rad(np.concatenate([robot['real_left_deg'], robot['real_right_deg']], axis=1)).astype(np.float32)
    joints = np.concatenate([measured[ri], hands['left_rad'][hi], hands['right_rad'][hi]], axis=1).astype(np.float32)
    if joints.shape != (n, 58) or not np.isfinite(joints).all():
        raise ValueError(f'Missing/nonfinite measured joints: {raw}')
    flags = np.zeros(n, dtype=bool)
    flags[:n-32] = np.convolve(valid.astype(np.int64), np.ones(33, dtype=np.int64), mode='valid') == 33
    if not flags.any():
        raise ValueError(f'No valid 33-frame training windows: {raw}')
    return {'robot': robot, 'hands': hands, 'camera_times': camera_times, 'grid': grid, 'indices': indices,
            'errors': errors, 'valid': valid, 'flags': flags, 'joints': joints, 'manifest': manifest}


def encode_aligned(source, target, mapping, expected_frames):
    """Fully decode the raw video; encode only timestamp-selected RGB frames."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.partial.mp4')
    source_count = output_count = 0
    counts = np.bincount(mapping, minlength=expected_frames)
    with av.open(str(source)) as inp, av.open(str(temporary), 'w') as out:
        input_stream = inp.streams.video[0]
        input_stream.thread_count = 1
        stream = out.add_stream('libx264', rate=30)
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
                encoded.pts, encoded.time_base = output_count, Fraction(1, 30)
                for packet in stream.encode(encoded):
                    out.mux(packet)
                output_count += 1
        for packet in stream.encode():
            out.mux(packet)
    if source_count != expected_frames or output_count != len(mapping):
        raise ValueError(f'Decoded frame count mismatch: {source}')
    with av.open(str(temporary)) as check:
        if check.streams.video[0].frames != output_count:
            raise ValueError(f'Encoded frame count mismatch: {temporary}')
    temporary.replace(target)
    return {'source_decoded_frames': source_count, 'output_frames': output_count, 'width': stream.width, 'height': stream.height}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    download = json.loads((root / 'download_manifest_latest.json').read_text())
    if not download['complete']:
        raise ValueError('Download has not completed')
    count = int(download['episodes'])
    previous = root / 'lerobot_next_state_v3'
    old_alignment = root / 'aligned_native_v2'
    native = root / f'lerobot_next_state_v3_{count}'
    aligned = root / f'aligned_native_v2_{count}'
    if native.exists() or aligned.exists():
        raise FileExistsError('Refusing to replace an existing incremental preparation')
    native.mkdir(); (aligned / 'episodes').mkdir(parents=True)
    info = json.loads((previous / 'meta/info.json').read_text())
    hand_names = [name.removeprefix('left_hand_joint_') for name in info['features']['action']['names'][14:36]]
    old_table = pq.read_table(previous / 'meta/episodes/chunk-000/file-000.parquet')
    episodes = old_table.to_pylist()
    old_count = len(episodes)
    # Reproduce the previous independently audited alignment before extending it.
    golden = align_episode(root / 'episode_badminton/episode_0001', hand_names)
    with np.load(old_alignment / 'episodes/episode_0001.npz') as f:
        np.testing.assert_allclose(golden['grid'], f['aligned/monotonic_time_ns'], atol=1, rtol=0)
        np.testing.assert_array_equal(golden['indices'][0], f['aligned/robot_row_index'])
        np.testing.assert_array_equal(np.stack(golden['indices'][1:4], axis=1), f['aligned/camera_frame_indices'])
        np.testing.assert_array_equal(golden['valid'], f['aligned/time_valid'])
        np.testing.assert_array_equal(golden['indices'][-1], f['aligned/hand_row_index'])
    print('Existing-episode alignment regression passed', flush=True)
    for path in previous.glob('data/**/*.parquet'):
        dest = native / path.relative_to(previous);dest.parent.mkdir(parents=True, exist_ok=True);os.link(path, dest)
    for path in previous.glob('videos/**/*.mp4'):
        dest = native / path.relative_to(previous);dest.parent.mkdir(parents=True, exist_ok=True);os.link(path, dest)
    for path in (old_alignment / 'episodes').iterdir():
        if path.suffix in ('.json', '.npz'):
            os.link(path, aligned / 'episodes' / path.name)
    source_schema = pq.read_schema(next(previous.glob('data/**/*.parquet')))

    def prepare(ep_number):
        raw = root / 'episode_badminton' / f'episode_{ep_number:04d}'
        data = align_episode(raw, hand_names)
        ep = ep_number-1
        grid, indices, joints = data['grid'], data['indices'], data['joints']
        n = len(grid)
        ri, hi = indices[0], indices[-1]
        valid = data['valid']
        action_valid = valid & np.r_[valid[1:], False]
        action = np.concatenate([joints[1:], joints[-1:]])
        robot = data['robot']
        columns = {
            'action': action.tolist(), 'observation.state': joints.tolist(), 'timestamp': np.arange(n)/30,
            'frame_index': np.arange(n, dtype=np.int64), 'episode_index': np.full(n, ep, dtype=np.int64),
            'index': np.zeros(n, dtype=np.int64), 'task_index': np.zeros(n, dtype=np.int64),
            'annotation.time_valid': valid, 'annotation.action_valid': action_valid,
            'action_mask': np.broadcast_to(action_valid[:, None], (n, 58)).tolist(),
            'annotation.valid_window_start_33': data['flags'], 'source.monotonic_time_ns': grid,
            'source.robot_row_index': ri, 'source.arm_joint_command': robot['command_rad'][ri].tolist(),
            'source.time_error_ns': data['errors'].tolist(), 'source.hand_row_index': hi,
            'observation.sim_tcp_left_pose': np.concatenate([robot['sim_tcp_left_pos'][ri], robot['sim_tcp_left_quat'][ri]], axis=1).tolist(),
            'observation.sim_tcp_right_pose': np.concatenate([robot['sim_tcp_right_pos'][ri], robot['sim_tcp_right_quat'][ri]], axis=1).tolist(),
            **{f'source.{camera}.frame_index': index for camera, index in zip(CAMERAS, indices[1:4])},
        }
        # Global row indices are filled once episode lengths are known, in order.
        table = pa.Table.from_pydict(columns, schema=source_schema)
        rel = info['data_path'].format(chunk_index=0, file_index=ep)
        dest = native / rel;dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest, compression='zstd')
        ep_row = {'episode_index': ep, 'length': n, 'tasks': ['badminton'],
                  'dataset_from_index': 0, 'dataset_to_index': n, 'data/chunk_index': 0, 'data/file_index': ep,
                  'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
                  'valid_windows_33': int(data['flags'].sum())}
        for feature, values, mask in [('action', action, action_valid), ('observation.state', joints, valid)]:
            x = values[mask].astype(np.float64)
            for key,value in [('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)), ('std', x.std(0)), ('count', np.array([len(x)]))]:
                ep_row[f'stats/{feature}/{key}'] = value.tolist()
        camera_report = {}
        for camera, index in zip(CAMERAS, indices[1:4]):
            key = VIDEO_KEYS[camera]
            video_rel = info['video_path'].format(video_key=key, chunk_index=0, file_index=ep)
            camera_report[camera] = encode_aligned(raw/camera/'color_take001.mp4', native/video_rel, index, len(data['camera_times'][camera]))
            camera_report[camera]['source_sha256'] = sha256(raw/camera/'color_take001.mp4')
            camera_report[camera]['source_path'] = str(raw/camera/'color_take001.mp4')
            camera_report[camera]['timestamps_sha256'] = sha256(raw/camera/'timestamps_take001.jsonl')
            ep_row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': ep,
                           f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': n/30})
        archive = {f'robot/{key}': value for key,value in robot.items()}
        archive.update({f'hands/{key}': value for key,value in data['hands'].items()})
        archive.update({f'cameras/{camera}/monotonic_time_ns': times for camera,times in data['camera_times'].items()})
        archive.update({'aligned/monotonic_time_ns': grid, 'aligned/robot_row_index': ri,
                        'aligned/camera_frame_indices': np.stack(indices[1:4], axis=1), 'aligned/hand_row_index': hi,
                        'aligned/time_error_ns': data['errors'], 'aligned/time_valid': valid,
                        'aligned/valid_window_starts_33': np.flatnonzero(data['flags'])})
        np.savez_compressed(aligned/'episodes'/f'episode_{ep_number:04d}.npz', **archive)
        report = {'episode_index': ep, 'source_name': raw.name, 'source_directory': str(raw),
                  'source_robot_sha256': sha256(raw/'robot_states.h5'), 'archive': f'episodes/episode_{ep_number:04d}.npz',
                  'aligned_rows': n, 'valid_rows': int(valid.sum()), 'invalid_rows': int((~valid).sum()),
                  'valid_windows_33': int(data['flags'].sum()), 'maximum_alignment_error_ms': dict(zip(['robot',*CAMERAS,'hands'], (np.abs(data['errors']).max(0)/1e6).tolist())),
                  'source_manifest_episode': data['manifest']['episode'], 'source_recording_errors': [],
                  'recorded_real_tcp_auxiliary': sorted(key for key in robot if key.startswith('real_tcp_')),
                  'cameras': camera_report}
        write_json(aligned/'episodes'/f'episode_{ep_number:04d}.json', report)
        return ep_row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, ep): ep for ep in range(old_count+1, count+1)}
        for future in as_completed(futures):
            row = future.result();episodes.append(row)
            print(f'Prepared episode {row["episode_index"]+1}: {row["length"]} rows, {row["valid_windows_33"]} windows', flush=True)
    episodes.sort(key=lambda row: row['episode_index'])
    offset = 0
    for row in episodes:
        ep = row['episode_index']
        row['dataset_from_index'], row['dataset_to_index'] = offset, offset+row['length']
        if ep >= old_count:
            path = native/info['data_path'].format(chunk_index=0, file_index=ep)
            table = pq.read_table(path)
            table = table.set_column(table.column_names.index('index'), 'index', pa.array(np.arange(offset, offset+len(table)), type=pa.int64()))
            pq.write_table(table, path, compression='zstd')
        offset += row['length']
    info.update(total_episodes=count, total_frames=offset, splits={'train': f'0:{count}'})
    meta = native/'meta/episodes/chunk-000';meta.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes, schema=old_table.schema), meta/'file-000.parquet')
    os.link(previous/'meta/tasks.parquet', native/'meta/tasks.parquet')
    write_json(native/'meta/info.json', info)
    # Native moments use all valid demonstrations, before FK conversion.
    aggregate = {feature: [] for feature in ['action', 'observation.state']}
    for row in episodes:
        path = native/info['data_path'].format(chunk_index=0, file_index=row['episode_index'])
        t = pq.read_table(path)
        for feature, valid_col in [('action', 'annotation.action_valid'), ('observation.state', 'annotation.time_valid')]:
            aggregate[feature].append(np.stack(t[feature].to_numpy())[t[valid_col].to_numpy()])
    stats = {}
    for feature, arrays in aggregate.items():
        x = np.concatenate(arrays).astype(np.float64)
        stats[feature] = {key:value.tolist() for key,value in [('min', x.min(0)), ('max', x.max(0)), ('mean', x.mean(0)), ('std', x.std(0)), ('count', np.array([len(x)]))]}
    write_json(native/'meta/stats.json', stats)
    report = {'complete': True, 'total_episodes': count, 'new_episodes': count-old_count, 'total_rows': offset,
              'valid_windows_33': sum(row['valid_windows_33'] for row in episodes), 'source_revision': download['revision'],
              'raw_directory': str(root/'episode_badminton'), 'previous_native': str(previous),
              'source_alignment': str(aligned), 'fps': 30, 'max_time_error_ms': 50,
              'alignment_regression_verified': True, 'new_raw_videos_fully_decoded': (count-old_count)*3,
              'new_real_tcp_fields': 'preserved in raw HDF5 and aligned NPZ; not substituted for uniform URDF FK'}
    write_json(aligned/'audit_report.json', report)
    write_json(native/'conversion_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
