#!/bin/bash
# Autonomous queue: wait for v5 to finish, then run v6, v7, update charts.
# One heavy job at a time.
set -e
cd /Users/saiduttaabhishekdash/gemma-ws-edit

# Wait for v5 pid 37237 equivalent (any run_full_bench)
while pgrep -f "run_full_bench.py" > /dev/null; do sleep 60; done
echo "v5 done, starting v6" >> logs/queue.log

.venv/bin/python scripts/run_full_bench.py --model models/gemma-3-270m-math-v6-rome-mlp-a25 --outdir results/full_v6_rome_mlp_a25 --tag v6_rome_mlp_a25 > logs/full_v6.out 2>&1
echo "v6 done, starting v7" >> logs/queue.log

.venv/bin/python scripts/run_full_bench.py --model models/gemma-3-270m-math-v7-rome-mlp-late --outdir results/full_v7_rome_mlp_late --tag v7_rome_mlp_late > logs/full_v7.out 2>&1
echo "v7 done, updating charts" >> logs/queue.log

# Update benchmark_summary.csv from actual MPS JSONs (baseline, v3 best so far)
.venv/bin/python - << 'PY'
import json, glob
def load(pat):
    f=sorted(glob.glob(pat))[0]
    return json.load(open(f))["results"]
base=load("results/full_baseline_local/*/results_*.json")
v3=load("results/full_v3_rome_mlp/*/results_*.json")
rows=[]
rows.append(("GSM8K","gsm8k_cot","exact_match_flexible", base["gsm8k_cot"]["exact_match,flexible-extract"], v3["gsm8k_cot"]["exact_match,flexible-extract"], base["gsm8k_cot"]["exact_match_stderr,flexible-extract"], v3["gsm8k_cot"]["exact_match_stderr,flexible-extract"]))
rows.append(("GSM8K","gsm8k_cot","exact_match_strict", base["gsm8k_cot"]["exact_match,strict-match"], v3["gsm8k_cot"]["exact_match,strict-match"], base["gsm8k_cot"]["exact_match_stderr,strict-match"], v3["gsm8k_cot"]["exact_match_stderr,strict-match"]))
rows.append(("MATH-500","hendrycks_math500","exact_match", base["hendrycks_math500"]["exact_match,none"], v3["hendrycks_math500"]["exact_match,none"], base["hendrycks_math500"]["exact_match_stderr,none"], v3["hendrycks_math500"]["exact_match_stderr,none"]))
import csv
with open("charts/data/benchmark_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["benchmark","task","metric","baseline","edited","baseline_stderr","edited_stderr"]); w.writerows(rows)
print("CSV updated", rows)
PY

.venv/bin/python charts/generate_charts.py >> logs/charts.log 2>&1
echo "charts done" >> logs/queue.log
echo "AUTONOMOUS QUEUE COMPLETE" >> logs/queue.log
