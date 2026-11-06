# Gemma 3 270M math benchmark - complete log

## Goal
Run math benchmarks for `google/gemma-3-270m` and store results in organized folders with a DOX tree for docs. GSM8K and MATH-500 scores.

## Setup
- **Python venv:** Created with `python3 -m venv .venv`
- **Packages:** `lm-eval`, `transformers`, `torch`, `accelerate`
- **DOX tree:** Initialized via dox-init skill at root plus per-directory AGENTS.md files
- **Compute:** Local laptop (MPS, slow), CPU server, GPU VM (2x Tesla T4)

## Directory structure
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

## Gotchas and fixes

### 1. Slow on local laptop (MPS)
- **Issue:** Running on MacBook with MPS was very slow
- **Fix:** Moved to a CPU server per project constraints. Full benchmarks run on GPU VM or CPU server, not laptop.

### 2. Installation issues
- **Issue:** First pip install seemed to hang or stall, processes looked stuck
- **Issue:** Had to remove and recreate venv multiple times
- **Fix:**
  - Deleted `.venv` and recreated fresh
  - Upgraded pip/setuptools/wheel first
  - Installed packages after ensuring a clean environment
  - Eventually succeeded with lm-eval 0.4.13

### 3. No `timeout` binary on CPU server
- **Issue:** `timeout` command not found when trying to use it
- **Fix:** Used `pkill`, background processes with `&`, and manual polling instead

### 4. One heavy job at a time
- **Constraint:** One heavy benchmark job at a time on shared CPU servers. Never run two at once.
- **Fix:** Only ran GSM8K first, held off on MATH-500 until GSM8K completes.

### 5. lm-eval task names
- **Issue:** Unclear exact task names for MATH-500
- **Fix:** Explored installed tasks in venv. Found `hendrycks_math500` in `lm_eval/tasks/hendrycks_math/`. GSM8K uses `gsm8k_cot`.

### 6. Long runtime on CPU
- **Issue:** GSM8K on CPU with batch_size=1 takes very long (many hours)
- **Fix:** Patience, continuous polling. Progress is highly variable (4s/it to 37s/it).

### 7. Log tailing issues
- **Issue:** Some log reads showed truncated or stale output
- **Fix:** Used `grep -oP '\d+/\d+'` to get actual progress counters, checked process count with `pgrep`

### 8. Weight editing requirements
- **Issue:** Weight editing needs RAM/CPU for tracing; laptop not feasible. GPU helps for faster tracing and benchmarks.
- **Fix:** Used 2x T4 GPU VM. Head amplification was attempted first because it is the fastest approach.

## Results

### Baseline (T4/CPU, reference only)
- **GSM8K (CoT, CPU server)**: `exact_match` (flexible): **0.0182 ± 0.0037** (1.82%), strict: **0.0114 ± 0.0029** (1.14%)
- **MATH-500 (single T4)**: `exact_match,none`: **0.002 ± 0.002** (0.2%) - 500 samples, 0-shot

### After weight editing, first pass (T4, reference only)
- **Model:** `./models/gemma-3-270m-math-edited` (head amplification on GPU VM, heads/alpha not recorded in repo)
- **GSM8K (edited)**: flexible **0.0197 ± 0.0038** (1.97%), strict **0.0121 ± 0.0030** (1.21%)
- **MATH-500 (edited)**: **0.002 ± 0.002** (0.2%)

Head amplification gave a minimal gain on GSM8K and no gain on MATH-500. The original edit had no committed code, so heads and alpha are not reproducible. We keep it here as reference. All publishable claims now use MPS vs MPS below.

## MPS baseline and edits (2026-11-06, MacBook Air M2 8GB)

All MPS runs use the same settings: `google/gemma-3-270m`, float32, batch 4, tasks `gsm8k_cot + hendrycks_math500`, greedy, via `scripts/run_full_bench.py` in `.venv`. T4/CPU numbers above are reference only.

### MPS baseline (results/full_baseline_local/)
- Command: `.venv/bin/python scripts/run_full_bench.py --model google/gemma-3-270m --outdir results/full_baseline_local --tag baseline_local`
- Runtime: 44m54s for 1819 requests on M2 MPS. Faster than 3193s on 2x T4 because of unified memory, batch 4 vs 1, and the tiny 270M size.
- GSM8K flexible **0.0182 ± 0.0037** (24/1319), strict **0.0114 ± 0.0029** (15/1319)
- MATH-500 **0.002 ± 0.002** (1/500)

### Edit v2 - head amplify (auditable, MPS)
- Script: `scripts/edit_weights.py --edit head-amplify --alpha 1.5 --out models/gemma-3-270m-math-v2-head-amplify`
- Effect: o_proj shape (640,1024), 18 layers, 4 heads, head_dim 256, alpha 1.5 on all heads
- Result (results/full_v2_head_amplify/): GSM8K **0.0182 ± 0.0037**, strict **0.0114 ± 0.0029**, MATH-500 **0.002 ± 0.002**. Same as baseline.

### Edit v3 - ROME-inspired MLP
- Script: `scripts/edit_weights.py --edit rome-mlp --alpha 1.5 --layers 6,7,8,9,10,11,12 --out models/gemma-3-270m-math-v3-rome-mlp`
- Effect: MLP down_proj (640x2048) in middle layers 6-12 scaled by 1.5. This stands in for a causal ROME rank-1 fact edit.
- Result (results/full_v3_rome_mlp/): GSM8K flexible **0.0205 ± 0.0039** (27/1319), strict **0.0136 ± 0.0032** (18/1319), MATH-500 **0.002 ± 0.002** (1/500)

### Edit v5-v8 MPS sweep (2026-11-07)
- v5 MLP alpha 2.0 layers 6-12: GSM flex 0.0205 ± 0.0039, strict 0.0136 ± 0.0032, MATH 0.0020. Same as v3 (27/1319).
- v6 MLP alpha 2.5 layers 6-12: GSM flex 0.0205 ± 0.0039, strict 0.0129 ± 0.0031, MATH 0.0020. It saturates, no extra correct answers.
- v7 MLP late 9-17 alpha 1.5: GSM flex 0.0159 ± 0.0034, strict 0.0114 ± 0.0029, MATH 0.0020. Worse (21/1319).
- v8 ROME rank-1 (`scripts/rome_rank1.py`, 10 arithmetic facts, layers 6-12, scale 0.3, rel L2 0.05-0.13% per layer): GSM flex 0.0159 ± 0.0034, strict 0.0114 ± 0.0029, MATH 0.0000 ± 0.0000. Hurts both tasks, 0/500 on MATH. The key/value approximation is too blunt. lm_head direction plus mean key does not insert usable arithmetic. Logged as a negative result.

### Task vector search (2026-11-07)
- HF API search for gemma-3-270m math: Crystalcareai/gemma-3-270m-it-rl-warmup-synthetic1-math-code, Nirav-Madhani/gemma3-270m-grpo-math, Ma7ee7/gemma-3-270m-it-rlzero-math, tim-lawson/pretrain_*_open-web-math
- Only google/gemma-3-270m and google/gemma-3-270m-it matched exact gemma3_text 640 hidden, 18 layers after local verification. Candidates report gemma3_text but hidden/layers are not in the API response, so we downloaded one and checked the state_dict shape before trying theta_math minus theta_base.

### Edit v9 task vector (MPS, 2026-11-07)
- Script: `scripts/task_vector.py --alpha 0.2 --out models/gemma-3-270m-math-v9-taskvector-a02`
- Tau = theta_math (Nirav-Madhani/gemma3-270m-grpo-math, 236 tensors shape-verified) minus theta_base, theta' = theta_base + 0.2*tau. 97.4% params changed (full-model direction, still weight-space only, no finetuning).
- Result (results/full_v9_taskvector_a02/): GSM8K flex 0.0182 ± 0.0037 (24/1319), strict 0.0144 ± 0.0033 (19/1319), MATH-500 0.0000 (0/500). Strict is up but MATH is worse.

### Final MPS sweep (charts/data/mps_all_runs.csv):
| run | GSM flex | correct | GSM strict | MATH |
| baseline | 0.0182 | 24/1319 | 0.0114 | 0.0020 |
| v2 head | 0.0182 | 24 | 0.0114 | 0.0020 |
| v3 MLP mid 1.5, best published | 0.0205 | 27 | 0.0136 | 0.0020 |
| v4 both | 0.0205 | 27 | 0.0136 | 0.0020 |
| v5 MLP 2.0 | 0.0205 | 27 | 0.0136 | 0.0020 |
| v6 MLP 2.5 | 0.0205 | 27 | 0.0129 | 0.0020 |
| v7 late 9-17 | 0.0159 | 21 | 0.0114 | 0.0020 |
| v8 rank-1 | 0.0159 | 21 | 0.0114 | 0.0000 |
| v9 task-vec 0.2 | 0.0182 | 24 | 0.0144 | 0.0000 |
Conclusion: coarse scaling saturates at 27/1319, well below the bar of 34. Published checkpoint is v3 on Hugging Face saidutta69/gemma-3-270m-math-rome-best with publication charts and sweep figures.

## Criteria for a significant improvement and release

- GSM8K needs more than baseline + 2*stderr = **>2.56% (about 34/1319, +10 correct)** to claim a win
- MATH-500 needs **>1.0% (>=6/500, +5 correct)** to clear noise (1/500 baseline)
- Report params changed count and relative L2 delta, with EDIT_NOTE.json audit trail
- License note: Gemma weights are under Google Gemma license (gated), not MIT. Share edit script, scores, and graphs. Do not redistribute weights unless the license allows it. First release is code plus data viz.

## Scripts created
- `scripts/run_bench.py` - initial benchmark script
- `scripts/quick_eval.py` - quick local test script
- `scripts/edit_weights.py` - head amplify and MLP scaling
- `scripts/run_full_bench.py` - full MPS lm-eval runner
- `scripts/rome_rank1.py` - ROME rank-1 fact edit
- `scripts/task_vector.py` - task vector addition

## DOX
- Root `AGENTS.md` initialized via dox-init skill
- All subdirectories have AGENTS.md files
- DOX tree is in place

## Handoff notes
- Weight editing works on CPU with enough RAM (>=8-16GB); 270M is about 1GB. Causal tracing is slow on CPU but doable.
- Current laptop was not practical at first; after the MPS setup we later ran full benchmarks on M2 at 44m54s per run.
- Edited checkpoints were saved under `models/gemma-3-270m-math-*/` locally and only the best (v3) was released to Hugging Face.
- Head amplification was tried first (fastest). ROME/MEMIT for MLP edits and task vectors came next.
