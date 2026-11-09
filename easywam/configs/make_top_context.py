"""Encode the episode_top instruction into the EasyWAM context cache.

Mirrors official/make_context_v2.py: same Cosmos-Reason1 text encoder, same
Cosmos-Predict2.5 text projection, same 128x1024 bf16 payload shape.
"""
import os
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from model.backbone.cosmos25.cosmos_video_text_encoder import Cosmos25TextEncoder
from model.backbone.cosmos25.loader import load_cosmos25_text_projection, resolve_cosmos25_dit_path

TASK = "pull the string to spin the top on the table"
TEXT = ("A video recorded from a robot's point of view executing the following instruction: " + TASK)
OUT = Path(os.environ.get(
    "TOP_CONTEXT_OUT",
    "/datapool/lianjh/easywam/official/badminton/context/top_context.pt"))
CK = Path(os.environ.get("COSMOS_CKPT_DIR", "/datapool/lianjh/easywam/code/easywam/checkpoints"))
DEVICE = os.environ.get("TOP_CONTEXT_DEVICE", "musa:0")


def main():
    reason = CK / "Cosmos-Reason1-7B"
    predict = CK / "Cosmos-Predict2.5-2B"
    for path in (reason, predict):
        if not path.exists():
            raise SystemExit(f"Missing checkpoint: {path}")
    enc = Cosmos25TextEncoder.from_pretrained(
        str(reason), torch_dtype=torch.bfloat16, max_length=128).to(DEVICE).eval()
    proj = load_cosmos25_text_projection(
        resolve_cosmos25_dit_path(str(predict)), device=DEVICE, torch_dtype=torch.bfloat16)
    enc.set_projector(proj)
    with torch.no_grad():
        ctx, mask, _ = enc([TEXT], return_token_mask=True)
    ctx = ctx[0].detach().cpu()
    nonzero = (ctx.float().abs().sum(-1) > 0)
    print("TEXT", TEXT)
    print("shape", tuple(ctx.shape), "encoder_mask_true", int(mask[0].sum()),
          "nonzero_rows", int(nonzero.sum()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": TASK, "instruction_text": TEXT,
                "context": ctx, "context_mask": nonzero}, OUT)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
