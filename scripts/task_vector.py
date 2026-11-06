#!/usr/bin/env python3
"""Task vector addition v9: theta' = theta_base + alpha * (theta_math - theta_base)
Base: google/gemma-3-270m
Math: Nirav-Madhani/gemma3-270m-grpo-math (same 236 tensors, shapes verified)
Weight-space programming only, no finetuning.
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument("--base", default="google/gemma-3-270m")
p.add_argument("--math", default="Nirav-Madhani/gemma3-270m-grpo-math")
p.add_argument("--out", required=True)
p.add_argument("--alpha", type=float, default=0.2)
a=p.parse_args()

print(f"Loading base {a.base} and math {a.math}")
base=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
math=AutoModelForCausalLM.from_pretrained(a.math, torch_dtype=torch.float32)
tok=AutoTokenizer.from_pretrained(a.base)
print("Both loaded")

with torch.no_grad():
    base_sd=base.state_dict()
    math_sd=math.state_dict()
    assert set(base_sd.keys())==set(math_sd.keys()), "key mismatch"
    total_sq=0.0; tau_sq=0.0; changed=0; tot_params=0
    for k in base_sd.keys():
        tau=(math_sd[k].float()-base_sd[k].float())*a.alpha
        base_sd[k].add_(tau)
        tau_sq += (math_sd[k].float()-base_sd[k].float()).pow(2).sum().item()
        changed += (tau!=0).sum().item()
        tot_params += base_sd[k].numel()
    base.load_state_dict(base_sd)
    rel=(tau_sq**0.5)
    print(f"alpha {a.alpha} tau L2 {rel:.4f} params changed {changed}/{tot_params} ({changed/tot_params*100:.1f}%)")

os.makedirs(a.out, exist_ok=True)
base.save_pretrained(a.out)
tok.save_pretrained(a.out)
note={"base":a.base,"math":a.math,"out":a.out,"edit":"task-vector","alpha":a.alpha,"method":"theta_base + alpha*(theta_math-theta_base), weight-space only, no finetuning"}
with open(os.path.join(a.out,"EDIT_NOTE.json"),"w") as f: json.dump(note,f,indent=2)
print(f"Saved {a.out} note {note}")
