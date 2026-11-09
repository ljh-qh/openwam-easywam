# Upstream provenance

No upstream OpenWAM / EasyWAM source is vendored in this repository. This file
records which snapshots our code and configs were developed against.

| Component | Snapshot on Moore | Snapshot date | Upstream git |
|---|---|---|---|
| OpenWAM training stack | `/datapool/lianjh/openwam/code/source` | 2026-10-30 | not available (no `.git` in snapshot) |
| EasyWAM training stack (`official/`) | `/datapool/lianjh/easywam/official` | 2026-11-06 | not available (no `.git` in snapshot) |
| ActionDiT foundation (generic) | `/datapool/lianjh/easywam/official/ActionDiT_CosmosPredict25_2B_foundation.pt` | 2026-11-04 | built from Cosmos-Predict2.5; sha256 `1618d3f79c25cef0ac41a52c335392a27402841a185af668ce247f90a9a32737` |
| OpenWAM alpha foundation | `/datapool/lianjh/openwam/foundation/alpha_foundation` | 2026-10-30 | internal checkpoint |

Notes:

* Neither upstream snapshot kept its `.git` directory, so commit pins cannot be
  read from disk. Add the upstream URLs and commits here once confirmed.
* Everything in this repository is our own adapter/config/script code that sits
  on top of those snapshots; upstream packages are installed separately
  (see `env/`).
* The EasyWAM backbone configs reference the generic ActionDiT foundation, not
  the badminton/top fine-tuned weights.
