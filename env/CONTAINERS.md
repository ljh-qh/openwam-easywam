# Containers and runtime environment

All runs use Moore Threads MUSA GPUs (MTT S5000, 8 per node) inside Docker
containers that mount the shared `/datapool` volume.

## Images

| Purpose | Image |
|---|---|
| OpenWAM training | `lianjh/easywam-eval:snapshot` |
| EasyWAM training | `local/mate-fa3-torchcodec-deepspeed:0.2.5-musa5.2.0-torch2.9.1-ds0.19.3-s5000` |

## OpenWAM container (example: worker36090)

```bash
docker run -d --name lianjh-openwam-36090 \
  --device /dev/mtgpu.0 --device /dev/mtgpu.1 --device /dev/mtgpu.2 --device /dev/mtgpu.3 \
  --device /dev/mtgpu.4 --device /dev/mtgpu.5 --device /dev/mtgpu.6 --device /dev/mtgpu.7 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 --device /dev/dri/renderD130 \
  --device /dev/dri/renderD131 --device /dev/dri/renderD132 --device /dev/dri/renderD133 \
  --device /dev/dri/renderD134 --device /dev/dri/renderD135 \
  --shm-size 32g -v /datapool:/datapool lianjh/easywam-eval:snapshot sleep infinity

docker exec -d lianjh-openwam-36090 bash -lc \
  'source /datapool/lianjh/easywam/scripts/env_musa.sh; \
   cd /datapool/lianjh/openwam && bash scripts/run_bottle.sh > logs/bottle_train.log 2>&1'
```

## EasyWAM container (example: worker36096)

```bash
docker run -d --name lianjh-easywam-36096 \
  --device /dev/mtgpu.0 ... --device /dev/mtgpu.7 \
  --device /dev/dri/renderD128 ... --device /dev/dri/renderD135 \
  --shm-size 32g -v /datapool:/datapool \
  local/mate-fa3-torchcodec-deepspeed:0.2.5-musa5.2.0-torch2.9.1-ds0.19.3-s5000 sleep infinity
```

EasyWAM training environment (extra flags required on MUSA):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MCCL_IB_DISABLE=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/datapool/lianjh/easywam/envs/driver-libs:$LD_LIBRARY_PATH
export PYTHONPATH=/datapool/lianjh/openwam/work/easywam_deps:\
/datapool/lianjh/easywam/envs/musa/lib/python3.10/site-packages:\
/datapool/lianjh/easywam/official/code/src:\
/datapool/lianjh/easywam/code:/datapool/lianjh/easywam/code/vendor:\
/datapool/lianjh/easywam/code/vendor/openwam-source
```

`env_musa.sh` (copied next to this file) sets `PATH`/`PYTHONPATH` for the MUSA
env, `MCCL_IB_DISABLE=1`, `DS_ACCELERATOR=musa` and the driver-libs library
path. The EasyWAM container image does not ship `datasets`/`diffusers`, so those
live in the isolated directory above (see `pip-freeze-easywam.txt`).

## GPU preflight

`openwam/scripts/run_*.sh` check that 8 MUSA devices are visible and each has
more than 70 GiB free before launching; they also take a `flock` so duplicate
launches fail fast.
