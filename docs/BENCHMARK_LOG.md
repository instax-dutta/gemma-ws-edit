# Gemma 3 270M Math Benchmark - Complete Log

## Goal
Run math benchmarks for `google/gemma-3-270m` with results stored in organized directories, DOX tree for documentation. GSM8K and MATH-500 scores.

## Setup
- **Python venv:** Created with `python3 -m venv .venv`
- **Packages:** `lm-eval`, `transformers`, `torch`, `accelerate`
- **DOX tree:** Initialized via dox-init skill at root + per-directory AGENTS.md files
- **Compute environments:** Local laptop (MPS, slow), CPU server, GPU VM (2x Tesla T4)

## Directory Structure
```
gemma-ws-edit/
├── AGENTS.md (root DOX)
├── .venv/
├── docs/ (documentation)
├── dox/
├── logs/ (benchmark logs)
├── models/ (model cache if needed)
├── results/ (benchmark outputs)
└── scripts/ (run scripts)
```

## Gotchas & Fixes

### 1. Slow on local laptop (MPS)
- **Issue:** Running on MacBook with MPS was very slow
- **Fix:** Moved to a CPU server per project constraints. Full benchmarks run on GPU VM or CPU server, not laptop.

### 2. Installation issues
- **Issue:** First pip install seemed to hang/stall, processes appeared stuck
- **Issue:** Had to remove and recreate venv multiple times
- **Fix:**
  - Deleted `.venv` and recreated fresh
  - Upgraded pip/setuptools/wheel first
  - Installed packages after ensuring clean environment
  - Eventually succeeded with lm-eval 0.4.13

### 3. No `timeout` binary on CPU server
- **Issue:** `timeout` command not found when trying to use it
- **Fix:** Used `pkill`, background processes with `&`, and manual polling instead

### 4. One heavy job at a time
- **Constraint:** One heavy benchmark job at a time on shared CPU servers. Never run two concurrently.
- **Fix:** Only ran GSM8K first, held off on MATH-500 until GSM8K completes.

### 5. lm-eval task names
- **Issue:** Unclear exact task names for MATH-500
- **Fix:** Explored installed tasks in venv. Found `hendrycks_math500` in `lm_eval/tasks/hendrycks_math/`. GSM8K uses `gsm8k_cot`.

### 6. Long runtime on CPU
- **Issue:** GSM8K on CPU with batch_size=1 takes very long (many hours)
- **Fix:** Patience, continuous polling. Progress is highly variable (4s/it to 37s/it).

### 7. Log tailing issues
- **Issue:** Some log reads showed truncated/stale output
- **Fix:** Used `grep -oP '\d+/\d+'` to get actual progress counters, checked process count with `pgrep`

### 8. Weight editing requirements
- **Issue:** Weight editing needs RAM/CPU for tracing; laptop not feasible. GPU helpful for faster tracing/benchmarks.
- **Fix:** Used 2x T4 GPU VM. Head amplification was attempted (fastest approach).

## Results

### Baseline
- **GSM8K (CoT, CPU server)**: `exact_match` (flexible): **0.0182 ± 0.0037** (1.82%), strict: **0.0114 ± 0.0029** (1.14%)
- **MATH-500 (single T4)**: `exact_match,none`: **0.002 ± 0.002** (0.2%) - 500 samples, 0-shot

### After weight editing (head amplification, first pass)
- **Model**: `./models/gemma-3-270m-math-edited` (head amplification applied on GPU VM)
- **GSM8K (edited)**: flexible **0.0197 ± 0.0038** (1.97%), strict **0.0121 ± 0.0030** (1.21%) - small improvement (+0.15pp flexible)
- **MATH-500 (edited)**: **0.002 ± 0.002** (0.2%) - no change from baseline

**Notes:** Head amplification gave minimal gain on GSM8K, no improvement on MATH-500. This was the first pass. More targeted approaches (ROME-style MLP edits, circuit analysis) may be needed for larger gains.

## Scripts Created
- `scripts/run_bench.py` - initial benchmark script
- `scripts/quick_eval.py` - quick local test script

## DOX
- Root `AGENTS.md` initialized via dox-init skill
- All subdirectories have AGENTS.md files
- DOX tree is in place

## Handoff Notes
- Weight editing feasible on CPU with enough RAM (>=8-16GB); 270M ~1GB. Causal tracing slow on CPU but doable.
- Current laptop not feasible; 2x T4 VM worked well for editing and benchmarking.
- Edited checkpoint saved at `models/gemma-3-270m-math-edited/` on GPU VM.
- Results saved on GPU VM.
- Head amplification was tried first (fastest). Consider ROME/MEMIT for MLP edits or task vectors if compatible math-tuned weights available.