#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime

# Run with lm_eval CLI via subprocess for simplicity
import subprocess

model = "google/gemma-3-270m"
outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
logdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(outdir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(logdir, f"bench_{timestamp}.log")

tasks = ["gsm8k_cot", "minerva_math", "math"]  # common math tasks

cmd = [
    sys.executable, "-m", "lm_eval",
    "run",
    "--model", "hf",
    "--model_args", f"pretrained={model},dtype=float32",
    "--tasks", ",".join(tasks),
    "--device", "mps" if sys.platform == "darwin" else "cpu",
    "--batch_size", "1",
    "--output_path", outdir,
    "--log_samples",
]

with open(logfile, "w") as f:
    f.write(f"Command: {' '.join(cmd)}\n")
    f.write(f"Timestamp: {timestamp}\n\n")

with open(logfile, "a") as f:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        f.write(line)
    proc.wait()

with open(logfile, "a") as f:
    f.write(f"\n\nExit code: {proc.returncode}\n")
