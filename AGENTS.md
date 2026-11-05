# gemma-ws-edit - Direct Weight-Space Editing for Gemma 3 270M Math Reasoning

## Purpose
Research project exploring whether direct weight-space programming (no finetuning/retraining) can improve math reasoning in Google's Gemma 3 270M model. Evaluated on GSM8K (CoT) and MATH-500 benchmarks. Every edit is a weight manipulation only -- reversible, minimal, compared against a documented baseline.

## Ownership
- Project: gemma-ws-edit
- Model: google/gemma-3-270m
- Benchmarks: GSM8K (CoT), MATH-500 (hendrycks_math500)

## Structure
- `scripts/` - Benchmark execution scripts (lm-eval, custom runners)
- `results/` - Evaluation results (JSON, JSONL samples)
- `logs/` - Execution logs from runs on different machines
- `models/` - Model artifacts and checkpoints (large files, not committed)
- `docs/` - Documentation, research notes, benchmark logs
- `dox/` - DOX tree documentation

## Baseline Results
- **GSM8K (CPU server)**: exact_match (flexible) = 0.0182 ± 0.0037 (1.82%), strict = 0.0114 ± 0.0029 (1.14%)
- **MATH-500 (single T4)**: exact_match = 0.002 ± 0.002 (0.2%)

## First-Pass Results (head amplification)
- **GSM8K (edited)**: flexible 0.0197 ± 0.0038 (1.97%), strict 0.0121 ± 0.0030 (1.21%) - +0.15pp flexible
- **MATH-500 (edited)**: 0.002 ± 0.002 (0.2%) - no change

## Current Goal
Use weight-space programming (direct weight edits: attention head amplification, ROME-style MLP fact edits, task vector application if available) to improve scores on GSM8K and MATH-500 without finetuning or retraining. Edits performed on suitable hardware (CPU server/GPU VM). Benchmarks re-run after edits for fair comparison.

## Child DOX Index
- `scripts/` - Benchmark execution scripts
- `results/` - Evaluation results and samples
- `logs/` - Execution logs
- `models/` - Model storage
- `docs/` - Documentation and benchmark logs
- `dox/` - DOX tree documentation