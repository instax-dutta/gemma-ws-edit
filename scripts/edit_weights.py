#!/usr/bin/env python3
"""Weight-space edit for Gemma 3 270M math -- no finetuning.

Two edits, both reversible weight scalings saved as new checkpoints:
1) head-amplify: scale o_proj slices for selected heads by alpha.
   Reproduces Edit-1 in an auditable way (first edit had no code in repo).
   Default: all 18 layers x 4 heads, alpha=1.5 on o_proj rows for attn output.
   Gemma3: hidden=640, heads=4, head_dim=256, o_proj shape (640, 1024).
   o_proj input dim = 1024 = 4*256, so head h occupies columns [h*256:(h+1)*256].
2) rome-mlp: targeted MLP down_proj scale on middle layers (6-12) by alpha.
   Small-model stand-in for causal-tracing ROME when full tracing is deferred.
   MLP: down_proj shape (640, 2048). Scaling down_proj scales MLP output.

Usage:
  .venv/bin/python scripts/edit_weights.py --edit head-amplify --alpha 1.5 --out models/gemma-3-270m-math-v2-head-amplify
  .venv/bin/python scripts/edit_weights.py --edit rome-mlp --alpha 1.5 --layers 6,7,8,9,10,11,12 --out models/gemma-3-270m-math-v3-rome-mlp
  .venv/bin/python scripts/edit_weights.py --edit both --alpha 1.5 --layers 6,7,8,9,10,11,12 --out models/gemma-3-270m-math-v4-both

Base defaults to google/gemma-3-270m (cached), device CPU for editing (fast, no OOM).
Each checkpoint dir gets EDIT_NOTE.json with edit, alpha, layers, base.
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="google/gemma-3-270m")
p.add_argument("--out", required=True)
p.add_argument("--edit", choices=["head-amplify", "rome-mlp", "both"], default="head-amplify")
p.add_argument("--alpha", type=float, default=1.5)
p.add_argument("--layers", type=str, default="6,7,8,9,10,11,12",
               help="Comma list for rome-mlp layers (default middle 6-12)")
p.add_argument("--heads", type=str, default="0,1,2,3",
               help="Comma list of heads to amplify (default all 4)")
a = p.parse_args()
layers = [int(x) for x in a.layers.split(",") if x != ""]
heads = [int(x) for x in a.heads.split(",") if x != ""]
head_dim = 256

print(f"Loading base: {a.base}")
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(a.base)
print(f"layers={model.config.num_hidden_layers} heads={model.config.num_attention_heads}")

def apply_head_amplify(alpha: float, heads: list[int]):
    for li, layer in enumerate(model.model.layers):
        w = layer.self_attn.o_proj.weight.data  # (640, 1024)
        assert w.shape == (640, 1024), w.shape
        for h in heads:
            s, e = h * head_dim, (h + 1) * head_dim
            w[:, s:e] *= alpha

def apply_rome_mlp(alpha: float, layer_ids: list[int]):
    for li in layer_ids:
        w = model.model.layers[li].mlp.down_proj.weight.data  # (640, 2048)
        assert w.shape[0] == 640, w.shape
        w *= alpha

if a.edit in ("head-amplify", "both"):
    print(f"Applying head-amplify alpha={a.alpha} heads={heads}")
    apply_head_amplify(a.alpha, heads)
if a.edit in ("rome-mlp", "both"):
    print(f"Applying rome-mlp alpha={a.alpha} layers={layers}")
    apply_rome_mlp(a.alpha, layers)

os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
note = {"base": a.base, "out": a.out, "edit": a.edit, "alpha": a.alpha,
        "layers": layers, "heads": heads, "head_dim": head_dim,
        "method": "weight scaling only, no finetuning"}
with open(os.path.join(a.out, "EDIT_NOTE.json"), "w") as f:
    json.dump(note, f, indent=2)
print(f"Saved edited checkpoint to {a.out}")
print(json.dumps(note, indent=2))
