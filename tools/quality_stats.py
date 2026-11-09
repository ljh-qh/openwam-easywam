"""Per-episode quality statistics for a 58-D native LeRobot v3 dataset.

Read-only: writes meta/quality.json and meta/quality_summary.json, never
modifies training data. Works for any of the OpenWAM task datasets (top/bottle).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FPS_DEFAULT = 30
PAUSE_THRESHOLD_DEFAULT = 0.05  # rad/s
LEFT_ARM = slice(0, 7)
RIGHT_ARM = slice(7, 14)
HANDS = slice(14, 58)


def episode_metrics(state, fps, pause_threshold):
    """state: (n, 58) joints in rad. Returns metric dict."""
    arms = state[:, LEFT_ARM.start:RIGHT_ARM.stop].astype(np.float64)
    hands = state[:, HANDS.start:HANDS.stop].astype(np.float64)
    dt = 1.0 / fps
    n = arms.shape[0]
    if n < 4:
        return None
    arm_vel = np.diff(arms, axis=0) / dt
    speed = np.linalg.norm(arm_vel, axis=1)
    accel = np.diff(arm_vel, axis=0) / dt
    jerk = np.diff(accel, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)
    left_speed = np.linalg.norm(arm_vel[:, LEFT_ARM], axis=1)
    right_speed = np.linalg.norm(arm_vel[:, RIGHT_ARM], axis=1)
    left_motion = float(left_speed.sum())
    right_motion = float(right_speed.sum())
    total_motion = left_motion + right_motion
    hands_vel = np.diff(hands, axis=0) / dt
    hand_activity = float(np.linalg.norm(hands_vel, axis=1).mean())

    path_len = float(np.linalg.norm(np.diff(arms, axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(arms[-1] - arms[0]))
    path_ratio = path_len / displacement if displacement > 1e-9 else float("inf")

    left_share = left_motion / total_motion if total_motion > 1e-9 else 0.5
    if left_share >= 0.6:
        dominant = "left"
    elif left_share <= 0.4:
        dominant = "right"
    else:
        dominant = "balanced"

    return {
        "frames": int(n),
        "duration_s": round(n / fps, 4),
        "arm_speed_mean": round(float(speed.mean()), 6),
        "arm_speed_p95": round(float(np.percentile(speed, 95)), 6),
        "arm_speed_max": round(float(speed.max()), 6),
        "pause_ratio": round(float((speed < pause_threshold).mean()), 6),
        "jerk_mean": round(float(jerk_mag.mean()), 4),
        "path_len": round(path_len, 6),
        "displacement": round(displacement, 6),
        "path_ratio": round(path_ratio, 6) if np.isfinite(path_ratio) else None,
        "left_motion_share": round(left_share, 4),
        "dominant_side": dominant,
        "hand_activity": round(hand_activity, 6),
    }


def quantiles(values, qs=(0, 5, 50, 90, 95, 100)):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not arr.size:
        return {}
    return {f"p{q}": round(float(np.percentile(arr, q)), 6) for q in qs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', required=True)
    parser.add_argument('--fps', type=int, default=FPS_DEFAULT)
    parser.add_argument('--pause-threshold', type=float, default=PAUSE_THRESHOLD_DEFAULT)
    args = parser.parse_args()

    root = Path(args.native)
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()

    records = []
    for row in sorted(episodes, key=lambda r: int(r['episode_index'])):
        ep = int(row['episode_index'])
        table = pq.read_table(root / f'data/chunk-000/file-{ep:03d}.parquet',
                              columns=['observation.state'])
        state = np.stack(table['observation.state'].to_numpy()).astype(np.float64)
        metrics = episode_metrics(state, args.fps, args.pause_threshold)
        if metrics is None:
            continue
        metrics.update(episode_index=ep, valid_windows_33=int(row['valid_windows_33']),
                       source=Path(row.get('source_name', '')).name if row.get('source_name') else None)
        records.append(metrics)

    durations = [r['duration_s'] for r in records]
    pauses = [r['pause_ratio'] for r in records]
    ratios = [r['path_ratio'] for r in records]
    dur_p95 = float(np.percentile(durations, 95))
    pause_p90 = float(np.percentile(pauses, 90))
    ratio_p95 = float(np.percentile(ratios, 95))

    for r in records:
        flags = []
        if r['duration_s'] > dur_p95:
            flags.append('duration_gt_p95')
        if r['pause_ratio'] > pause_p90:
            flags.append('pause_ratio_gt_p90')
        if r['path_ratio'] is not None and r['path_ratio'] > ratio_p95:
            flags.append('path_ratio_gt_p95')
        r['outlier_flags'] = flags

    payload = {
        'dataset': str(root),
        'units': {'arm_speed': 'rad/s', 'jerk': 'rad/s^3', 'path_len': 'rad', 'displacement': 'rad',
                  'hand_activity': 'rad/s', 'pause_threshold': 'rad/s'},
        'thresholds': {'pause_arm_speed_rad_s': args.pause_threshold,
                       'outlier_duration_p95_s': round(dur_p95, 4),
                       'outlier_pause_ratio_p90': round(pause_p90, 6),
                       'outlier_path_ratio_p95': round(ratio_p95, 6)},
        'fps': args.fps,
        'episodes': records,
    }
    (root / 'meta/quality.json').write_text(json.dumps(payload, indent=2) + '\n')

    summary = {
        'dataset': str(root),
        'episodes': len(records),
        'total_frames': int(sum(r['frames'] for r in records)),
        'total_duration_s': round(sum(r['duration_s'] for r in records), 3),
        'total_valid_windows_33': int(sum(r['valid_windows_33'] for r in records)),
        'quantiles': {
            'duration_s': quantiles(durations),
            'arm_speed_mean': quantiles([r['arm_speed_mean'] for r in records]),
            'arm_speed_p95': quantiles([r['arm_speed_p95'] for r in records]),
            'pause_ratio': quantiles(pauses),
            'jerk_mean': quantiles([r['jerk_mean'] for r in records]),
            'path_ratio': quantiles(ratios),
            'hand_activity': quantiles([r['hand_activity'] for r in records]),
        },
        'thresholds': payload['thresholds'],
        'side_stats': {
            'left': sum(1 for r in records if r['dominant_side'] == 'left'),
            'right': sum(1 for r in records if r['dominant_side'] == 'right'),
            'balanced': sum(1 for r in records if r['dominant_side'] == 'balanced'),
        },
        'outliers': [{'episode_index': r['episode_index'], 'flags': r['outlier_flags'],
                      'duration_s': r['duration_s'], 'pause_ratio': r['pause_ratio'],
                      'path_ratio': r['path_ratio']} for r in records if r['outlier_flags']],
        'long_tail_episodes': sorted(
            [r['episode_index'] for r in records if r['outlier_flags']]),
    }
    (root / 'meta/quality_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('QUALITY_OK', len(records), 'episodes ->', root / 'meta/quality.json')
    print('SUMMARY', summary['episodes'], summary['total_frames'], summary['total_duration_s'],
          'outliers', len(summary['outliers']))


if __name__ == '__main__':
    main()
