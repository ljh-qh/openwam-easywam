# OpenWAM × EasyWAM — manipulation training stack (badminton / top / bottle)

Code and configs we wrote for training state-supervised robot manipulation
policies on dual-arm Marvin hardware with Moore Threads (MUSA) GPUs, using two
upstream training stacks:

* **OpenWAM** — single-stage world-action model (`openwam/`), 80-D end-effector
  actions with 62 supervised dimensions.
* **EasyWAM** — two-stage video/action flow model (`easywam/`), DiT on top of a
  Cosmos-Predict2.5 foundation.

Three tasks are covered end-to-end: **badminton**, **top**, **bottle**.

## Layout

```
data_pipeline/     raw teleop -> aligned 58-D LeRobot v3 -> EEF80 + quality stats
openwam/readers/   dataset readers (58-D and hybrid)
openwam/configs/   training configs per task
openwam/scripts/   training entrypoints + launcher wrappers
easywam/adapters/  dataset adapters (raw -> model batches, prompt validation)
easywam/configs/   data/task/model configs + context builders
tools/             chunked ModelScope upload/download, verification, SSH helper
env/               env_musa template, pip freezes, container recipes
docs/              handoff notes
vendor/            upstream provenance (no upstream code vendored)
```

## Data pipeline

Raw recordings come from ModelScope dataset `jiawangli/physwam`:

| task | raw dirs | usable episodes | frames | valid 33-frame windows |
|---|---|---|---|---|
| top | 203 | 197 | 115,293 | 105,748 |
| bottle | 201 (194 unique) | 188 | 221,084 | 210,041 |

1. **Align** — `data_pipeline/align_episode_top.py` resamples the 30 Hz
   teleop streams by nearest neighbour (`|err| <= 50 ms`), keeps only valid
   33-frame windows, re-encodes videos so frame *k* equals row *k*, and writes a
   58-D native LeRobot v3 dataset (14 arm joints + 44 hand joints, radians).
2. **Convert** — `data_pipeline/convert_to_openwam.py` runs the Marvin URDF
   kinematics and emits 80-D end-effector actions (62 active dims).
3. **Quality** — `data_pipeline/quality_stats.py` writes `meta/quality.json`
   and `meta/quality_summary.json` (speed, pause ratio, jerk, path ratio,
   dominant side, hand activity, outlier flags). Read-only: never filters.

## Training

### OpenWAM (8×MUSA, bf16, ZeRO-2)

```bash
# inside lianjh-openwam-<node>, after `source env/env_musa.sh`
cd openwam
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --master_port=29631 scripts/train_bottle.py        # config-name bottle_state_5ep
# or the wrappers (GPU preflight + flock):
bash scripts/run_top.sh
bash scripts/run_bottle.sh
```

Hyperparameters (all tasks): batch 2/GPU × 8 ranks × grad-accum 8, lr 1e-4
cosine, 5% warmup, 5 epochs, save every 2000 steps, finetune from
`foundation/alpha_foundation`, resume disabled.

### EasyWAM (8×MUSA, accelerate + DeepSpeed ZeRO-2)

```bash
# inside lianjh-easywam-<node>, after env setup (see env/CONTAINERS.md)
cd easywam/official/code
python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 --main_process_port 29637 \
  scripts/train.py task=bottle_easywam_mot_joint_cosmos25
```

EasyWAM configs mirror OpenWAM: lr 1e-4, batch 2/GPU, grad-accum 8, cosine,
warmup 0.05, 5 epochs (`max_steps` 8205 for bottle), `save_every=1000`,
state-only supervision, generic ActionDiT foundation (no task-specific
pretraining).

## Weights and data on ModelScope

| artifact | repo | path |
|---|---|---|
| OpenWAM top (33,050) | `ljhqh24/openwam` | `checkpoints/top_state_5ep_33050/` |
| OpenWAM bottle (65,640) | `ljhqh24/openwam` | `checkpoints/bottle_state_5ep_65640/` |
| EasyWAM top (4,131) | `ljhqh24/easywam` | `top_5ep/` |
| EasyWAM bottle (8,205) | `ljhqh24/easywam` | `bottle_5ep/` |
| raw episodes (`episode_top`, `episode_bottle`) | `jiawangli/physwam` | dataset repo |

Large checkpoints are stored as verified 1 GiB parts plus
`upload_manifest.json` (per-part sha256 + total sha256); `tools/` contains the
4-way parallel uploader and the stream-merge downloader/verifier used to land
them on the 4090 deploy host.

## License / provenance

Upstream OpenWAM and EasyWAM code is **not** vendored here; see
`vendor/README.md` for the snapshot paths and dates used in our runs.
