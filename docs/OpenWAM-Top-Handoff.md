# OpenWAM episode_top（转陀螺）训练交接

日期：2026-11-04 01:20（Asia/Shanghai）
状态：**数据转换完成，8 卡训练已启动**（2026-11-04 01:15 起）

## 1. 任务指令

`pull the string to spin the top on the table`

- 写入 `meta/tasks.parquet`（task_index=0），训练 prompt 直接取自该表。
- `meta/conversion.json` 的 `task_instruction` 已同步改为该指令。

## 2. 数据链路

| 阶段 | 路径 | 规模 |
| --- | --- | --- |
| 原始 | `/datapool/lianjh/openwam_top/raw/episode_top` | 203 集，1.27 GB |
| 58 维 native | `/datapool/lianjh/openwam_top/data/lerobot_native_58` | 197 集 / 115,293 帧 / 1.7 GB |
| 对齐审计 | `/datapool/lianjh/openwam_top/data/aligned_native` | 197 套 npz+json，24 MB |
| EEF80 | `/datapool/lianjh/openwam_top/data/openwam_eef80` | 197 集 / 105,748 窗口 / 149 MB（视频为符号链接） |

- 对齐脚本：`/datapool/lianjh/openwam/code/badminton/align_episode_top.py`
  （工作副本 `/datapool/lianjh/openwam_top/work/align_top.py`）
- 30 Hz 最近邻网格，`|error| ≤ 50 ms`；`observation.state = 14 臂(rad) + 44 手(rad)`；
  `action[t] = state[t+1]`；`action_mask = action_valid`。
- 记录字段：`source.monotonic_time_ns`、`source.time_error_ns`(5 路)、
  `annotation.time_valid`、`annotation.action_valid`、`annotation.valid_window_start_33`、
  `source.robot_row_index`、`source.hand_row_index`、三路 `source.<cam>.frame_index`。
- 视频按时间戳重编码为 30 fps，保证第 k 帧 == 第 k 行。
- 训练只用实测状态；`command_rad` 仅作为 `source.arm_joint_command` 保留，不参与监督。

### 排除的 6 集（203-197）

- `episode_0051` / `0174` / `0175` / `0176` / `0199`：manifest `status=incomplete` + recording_errors
- `episode_0171`：缺 `robot_state/pico_teleop`

## 3. 训练配置

- 配置：`/datapool/lianjh/openwam/configs/top_state_5ep.yaml`
  （复制自 `hybrid_paraphrase_5ep.yaml`；`dataloader.type: badminton`，删除
  `hybrid_stats`/`paraphrases`，`dataset_dir` 指向新数据集，`output_path` 指向新 run）
- 入口：`/datapool/lianjh/openwam/scripts/train_top.py`（config-name `top_state_5ep`）
- 启动：`/datapool/lianjh/openwam/scripts/run_top.sh`（8 卡预检 + flock + torchrun）
- 预热权重：`/datapool/lianjh/openwam/alpha_foundation`（checkpoint_step_154000）
- `resume_ckpt_path: null`，`num_epochs: 5`，batch 2/卡，梯度累计 8（有效 batch 128）
- 输出：`/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56/`
  （容器时钟为 UTC，目录名比本地时间早 8 小时）

### 启动命令（在容器内）

```bash
docker exec -d lianjh-official-37009 bash -lc \
  'cd /datapool/lianjh/openwam && bash scripts/run_top.sh > /datapool/lianjh/openwam_top/logs/train_top_20261104_0112.log 2>&1'
```

- 日志：`/datapool/lianjh/openwam_top/logs/train_top_20261104_0112.log`
- 33050 micro-step（5 epoch × 6610），约 1.05 s/step，**预计 9–10 小时**（约 10:30–11:30 完成）
- 显存约 43.6 GB/卡（共 81.9 GB），8 卡利用率 81–100%

## 4. 启动前处理

- GPU0 原被旧的 `deploy.py`（step_35705、端口 8851、已运行 24h、无监听）占用 23.9 GB，
  已于 2026-11-04 01:06 用 SIGTERM 停止（pid 10900），GPU0 释放。
- 数据由容器 root 生成，已统一 `chown -R 1002:100 /datapool/lianjh/openwam_top/data`。

## 5. 巡检 / 停止

```bash
# 进度（该日志是 tqdm 回车流，建议这样取最后一条）
tail -c 2000 <log> | tr '\r' '\n' | tail -1
# GPU
mthreads-gmi | grep 'MiB('
# 进程
docker exec lianjh-official-37009 ps -eo pid,etime,args | grep -E 'train_top|torch.distributed' | grep -v grep
```

停止：`docker exec lianjh-official-37009 pkill -f train_top.py`（只杀本项目进程），
或对 torchrun 主进程发 SIGTERM。锁文件 `/datapool/lianjh/openwam/logs/top_training.lock`。

## 6. 注意

- 容器默认 root；跑 torch 不要用 `-u 1002`（无 passwd 条目，`getpass.getuser()` 会炸）。
- 容器时区 UTC，日志时间戳比本地早 8 小时。
- 旧 35705 权重同样是 state 监督，可直接做 A/B 对比。
- 堡垒机密码不入库；`~/.ssh/moore_hosts.conf` 指向 10.121.61.10。

## 7. 训练结果（2026-11-04 12:33 完成）

- 33050/33050，总耗时 **11:16:55**，最终 loss **0.0341**（video 0.0336 / action 0.0005），lr 收尾 1e-6。
- 8 卡已释放（0 MiB），进程退出，日志 0 条 Traceback/OOM。
- 产物（`runs/top_state_5ep/2026-11-03_17-15-56/`）：
  - `checkpoint_step_33050.safetensors`（24,813,767,464 B，final）
  - `checkpoint_step_32000.safetensors`（24,813,767,464 B）
  - `config.yaml`、`normalization_stats.npy`、`tokenizer/`
- 注意：`keep_last_k_ckpts: 2` 轮转了中间权重，`step_26000` 已被删除；`accel_state_*` 也被清理，
  因此**没有优化器 resume 状态**，只有推理权重。若要续训，只能以 `finetune_ckpt_path` 方式从 33050 再起步。

### A/B 可用权重

| 权重 | 路径 |
| --- | --- |
| 新 state 33050 | `/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56/checkpoint_step_33050.safetensors` |
| 新 state 32000 | `/datapool/lianjh/openwam_top/runs/top_state_5ep/2026-11-03_17-15-56/checkpoint_step_32000.safetensors` |
| 旧 35705 | `/datapool/lianjh/openwam/runs/hybrid_paraphrase_5ep/2026-11-02_02-40-37/checkpoint_step_35705.safetensors` |
