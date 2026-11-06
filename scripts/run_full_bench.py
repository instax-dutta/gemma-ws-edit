#!/usr/bin/env python3
"""Full lm-eval run: GSM8K CoT + MATH-500 on a given checkpoint.
Usage: .venv/bin/python scripts/run_full_bench.py --model <path-or-hf-id> --outdir results/full_<name> --tag <name> [--batch 4]
Logs to logs/bench_<tag>_<ts>.log, results to --outdir.
Matches repo baseline config: float32, tasks gsm8k_cot,hendrycks_math500, greedy.
"""
import argparse, subprocess, sys, os
from datetime import datetime

p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-3-270m")
p.add_argument("--outdir", required=True)
p.add_argument("--tag", default="run")
p.add_argument("--batch", default="4")
p.add_argument("--dtype", default="float32")
p.add_argument("--device", default="mps")
a = p.parse_args()

os.makedirs(a.outdir, exist_ok=True)
os.makedirs("logs", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"logs/bench_{a.tag}_{ts}.log"
cmd = [
    sys.executable, "-m", "lm_eval", "run",
    "--model", "hf",
    "--model_args", f"pretrained={a.model},dtype={a.dtype}",
    "--tasks", "gsm8k_cot,hendrycks_math500",
    "--device", a.device,
    "--batch_size", str(a.batch),
    "--output_path", a.outdir,
    "--log_samples",
]
print("CMD:", " ".join(cmd))
print("LOG:", logfile)
with open(logfile, "w") as f:
    f.write(f"Command: {' '.join(cmd)}\nTimestamp: {ts}\nModel: {a.model}\n")
print(f"Running now - progress in tail -f {logfile}", flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open(logfile, "a") as log:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    rc = proc.wait()
    log.write(f"\n\nExit code: {rc}\n")
print(f"Done exit={rc} results in {a.outdir} log in {logfile}")
sys.exit(rc)
