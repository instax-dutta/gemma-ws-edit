#!/usr/bin/env python3
"""Real-ish ROME rank-1 edit for Gemma 3 270M arithmetic facts.
Method (simplified ROME): For each fact (a op b = c), collect MLP key
activation in target layers via forward hooks on the prompt, then apply
rank-1 update to down_proj to push value direction toward correct answer token.

Facts (10): 2+3=5, 5+4=9, 6+7=13, 8+2=10, 3*4=12, 5*5=25, 6*2=12, 10-3=7, 15-6=9, 9-4=5
Layers: 6-12 (same MLP layers as v3). Scale: 0.3 per fact (small to avoid blowup).
Prompt format matches gsm8k_cot zero-shot: "Q: 2+3= ?"
This is still approximate (not full causal tracing Mediation analysis) but
inserts directional fact info, unlike pure scaling.
"""
import json, os, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FACTS = [("2+3","5"),("5+4","9"),("6+7","13"),("8+2","10"),("3*4","12"),("5*5","25"),("6*2","12"),("10-3","7"),("15-6","9"),("9-4","5")]

p=argparse.ArgumentParser()
p.add_argument("--base", default="google/gemma-3-270m")
p.add_argument("--out", default="models/gemma-3-270m-math-v8-rome-rank1")
p.add_argument("--scale", type=float, default=0.3)
p.add_argument("--layers", type=str, default="6,7,8,9,10,11,12")
a=p.parse_args()
layers=[int(x) for x in a.layers.split(",")]

print(f"Loading {a.base} on MPS for activation collect")
tok=AutoTokenizer.from_pretrained(a.base, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, device_map="mps", local_files_only=True)
model.eval()
lm_head = model.lm_head.weight.data  # (vocab, 640)
print("lm_head", lm_head.shape)

# Collect intermediate activations: register hooks on mlp.down_proj input (intermediate size 2048)
acts={li: [] for li in layers}
def make_hook(li):
    def hook(mod, inp, out):
        # inp[0]: (batch, seq, 2048)
        x=inp[0].detach().float()
        m=x.mean(dim=1).mean(dim=0)  # avg over seq and batch -> (2048,)
        acts[li].append(m.cpu())
    return hook

handles={}
for li in layers:
    handles[li]=model.model.layers[li].mlp.down_proj.register_forward_hook(make_hook(li))

# For each fact, forward prompt and collect key
fact_keys=[]
model.to("mps")
for expr, ans in FACTS:
    prompt=f"Q: {expr}= ?\nA:"
    inputs=tok(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        _=model(**inputs)
    # take last activation per layer as key for this fact
    keys={li: acts[li][-1] for li in layers}
    fact_keys.append((expr, ans, keys))
    print(f"collected {expr}={ans} key norm layer6 {keys[6].norm():.2f}")

for h in handles.values(): h.remove()

# Build rank-1 updates per layer
# value dir = lm_head row for answer token ("5" etc) normalized, minus mean over 0-9
# Use tokenizer to get single-token ids for answers (may be multi, use first)
ans_dirs=[]
for expr, ans in FACTS:
    ids=tok.encode(ans, add_special_tokens=False)
    tid=ids[0]
    vec=lm_head[tid].float().cpu()
    ans_dirs.append(vec / (vec.norm()+1e-8))

mean_dir=torch.stack(ans_dirs).mean(dim=0)

# Apply in CPU copy then save
print("Moving model to CPU for edit")
model.to("cpu")
for li in layers:
    W=model.model.layers[li].mlp.down_proj.weight.data  # (640,2048) on cpu
    delta=torch.zeros_like(W)
    for (expr, ans, keys), v in zip(fact_keys, ans_dirs):
        k=keys[li]  # (2048,)
        knorm2=(k**2).sum()+1e-8
        # rank-1 outer: (v - mean_dir) * k^T / ||k||^2 * scale
        v_adj=(v - mean_dir) * a.scale
        delta += torch.outer(v_adj, k) / knorm2
    W_new = W + delta
    # report relative size
    rel = delta.norm()/(W.norm()+1e-8)
    print(f"layer {li} delta rel L2 {rel.item():.5f} (scale {a.scale}) max abs {delta.abs().max():.5f}")
    model.model.layers[li].mlp.down_proj.weight.data = W_new

os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
note={"base":a.base,"out":a.out,"edit":"rome-rank1","method":"rank-1 outer product per fact on down_proj, keys from MPS activation mean, values from lm_head answer rows","facts":FACTS,"layers":layers,"scale":a.scale}
with open(os.path.join(a.out,"EDIT_NOTE.json"),"w") as f: json.dump(note,f,indent=2)
print(f"Saved {a.out}")
