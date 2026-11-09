# EasyWAM top 迁移交接（2026-11-04）

目标：在 worker37009 独立容器 `lianjh-easywam-37009` 里，用 `openwam_top` 的 58→EEF80 数据集
（197 集 / 105748 窗口，instruction `pull the string to spin the top on the table`）训练 state 监督的 EasyWAM。

## 运行状态

- 正式训练：2026-11-04 15:31 启动，8 卡，max_steps=4131（≈5 epoch），6.45 s/step，预计 ~7.4h。
- 日志：`/datapool/lianjh/openwam_top/logs/easywam_top_full_20261104_r2.log`（r1 为改显示前的短跑）
- 输出：`/datapool/lianjh/openwam_top/runs/easywam_top/`
- 冒烟：`logs/easywam_top_smoke.log`（2 卡 20 步，0 错）
- 显示约定：进度条/`[train] step` 按 micro-batch 计（optimizer step × grad_accum=8，总分母 33048），内部 4131 步不变；见 `official/code/src/trainer.py`（backup `.bak_microstep_20261104`）
- save_every=1000（内部步），约 4 个中间 checkpoint + final

## 启动命令

```bash
cd /datapool/lianjh/easywam/official/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       MCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
       LD_LIBRARY_PATH=/datapool/lianjh/easywam/envs/driver-libs:$LD_LIBRARY_PATH \
       PYTHONPATH=/datapool/lianjh/openwam_top/easywam_deps:/datapool/lianjh/easywam/envs/musa/lib/python3.10/site-packages:/datapool/lianjh/easywam/official/code/src:/datapool/lianjh/easywam/code:/datapool/lianjh/easywam/code/vendor:/datapool/lianjh/easywam/code/vendor/openwam-source
python -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 --main_process_port 29634 scripts/train.py \
  task=top_easywam_mot_joint_cosmos25 output_dir=/datapool/lianjh/openwam_top/runs/easywam_top
```

## 新建文件

- `code/top_adapter.py`：`TopEasyWAM`，EXPECTED_WINDOWS=105748，启动校验 prompt 与 context 一致，其余同 `badminton_adapter.py`（state[t+1]、62 ACTIVE、3×9×384×320）。
- `official/code/configs/data/top.yaml`
- `official/code/configs/task/top_easywam_mot_joint_cosmos25.yaml`（lr 1e-4、batch 2、accum 8、cosine、save_every 10000、max_steps 4131、output_dir 指向新目录、resume null）
- `official/code/configs/model/backbone/cosmos25_top.yaml`（badminton 版拷贝，仅 action_dit 路径不同）
- `official/code/configs/model/easywam_mot_joint_cosmos25_top.yaml`
- `official/make_top_context.py`
- `official/badminton/context/top_context.pt`（128×1024 bf16，mask 128，264,389 B）

原 badminton 文件未改；旧的 `badminton_top*` 中间命名已删除。

## 初始化权重

- `official/ActionDiT_CosmosPredict25_2B_top_foundation.pt`
  - 2,869,026,131 B
  - sha256 `1618d3f79c25cef0ac41a52c335392a27402841a185af668ce247f90a9a32737`
  - 由 `scripts/preprocess_action_dit_backbone.py` 从通用 Cosmos-Predict2.5-2B 生成（alpha scaling），不含 badminton 权重。
- 说明：archive 的 `ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt` 与当前 ActionDiT 结构不匹配（缺 `text_embedding.*`），不能直接用。

## 依赖

- `/datapool/lianjh/openwam_top/easywam_deps`：datasets 3.6.0、diffusers 0.31.0、einops 0.8.2、rich、multiprocess、dill、fsspec、xxhash。
- 复用 `/datapool/lianjh/easywam/envs/musa/.../site-packages`（pyarrow/av/pandas/hydra/omegaconf/transformers 4.57）。
- 全部 `pip --target`，未修改任何已有 env。

## 环境坑（重要）

1. 新容器必须带 `/dev/dri/renderD128..135`，否则 `torch.musa` 初始化失败。
2. 必须 `MCCL_IB_DISABLE=1` + `LD_LIBRARY_PATH` 含 `envs/driver-libs`（libmtml.so）；否则 MCCL all_reduce 报 `unhandled system error`。
3. 容器内没有 `accelerate` 命令，用 `python -m accelerate.commands.launch`。
4. Cosmos-Reason1-7B 已下载到 `archive/base_models/Cosmos-Reason1-7B`（hf-mirror，16GB），并软链到 `code/easywam/checkpoints/`。

## 检查点

- `save_every: 1000`：4131 步会存 1000/2000/3000/4000 + final 共 5 个 checkpoint。
- 旧 run（改显示前，~30 步）保留在 `runs/easywam_top_run1_20261104`。
