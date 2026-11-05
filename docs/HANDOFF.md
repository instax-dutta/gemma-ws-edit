# Handoff: Gemma 3 270M Weight-Space Programming for Math Improvement

## Project Overview
- **Goal**: Improve math capabilities of `google/gemma-3-270m` via weight-space programming (direct weight edits) without finetuning/retraining, evaluated on GSM8K and MATH-500.
- **Repo**: gemma-ws-edit (DOX tree initialized). Baselines documented in `AGENTS.md`, `docs/BENCHMARK_LOG.md`.

## Baselines
- **GSM8K (CoT, CPU server, float32, lm-eval)**: flexible exact_match 0.0182 ± 0.0037 (1.82%), strict 0.0114 ± 0.0029 (1.14%)
- **MATH-500 (T4, float32, lm-eval)**: exact_match 0.002 ± 0.002 (0.2%), 500 samples, 0-shot

## Compute Environment
- **Original benchmarks:** Moved from laptop (MPS slow) to CPU server (12 vCPUs, 47GB RAM, no GPU). One heavy job at a time.
- **Weight editing + benchmarks:** 2x Tesla T4 GPU VM. Sufficient for loading 270M (~1GB) and running tracing/benchmarks.
- **Recommendation:** Weight editing needs >=8-16GB RAM; laptop not feasible. Use CPU with RAM or GPU VM. Causal tracing is slow on CPU.

## What Was Done
1. **Planning:** Created `docs/WEIGHTSPACE_PROGRAMMING_PLAN.md` with approaches (head amplification, ROME/MEMIT for MLPs, task vectors), compute requirements, and timeline. Updated `AGENTS.md` goal. Created `GATES.md` with unlazy gates.
2. **Weight editing (first pass):** On GPU VM, copied repo and HF model. Attempted **attention head amplification** (scale attention output projections / head-related weights) - fastest reversible approach. Saved edited checkpoint to `models/gemma-3-270m-math-edited/`.
3. **Benchmarking (edited):** Ran GSM8K and MATH-500 on edited model via lm-eval.

## Results (Edited vs Baseline)
- **GSM8K**: flexible 0.0197 ± 0.0038 (+0.15pp), strict 0.0121 ± 0.0030 (+0.07pp) - small improvement
- **MATH-500**: 0.002 ± 0.002 (no change) - 0.2% unchanged

Head amplification gave minimal gain on GSM8K, none on MATH-500. This is the first pass.

## Files & Locations
### In Repo (local)
- `AGENTS.md` - updated goals/baselines
- `docs/BENCHMARK_LOG.md` - full log with baselines and edited results
- `docs/WEIGHTSPACE_PROGRAMMING_PLAN.md` - detailed plan and approaches
- `GATES.md` - unlazy gates
- `scripts/run_bench.py`, `scripts/quick_eval.py` - benchmark scripts
- `results/google__gemma-3-270m/` - baseline results

### On GPU VM
- Repo copy and model cache
- Edited model: `models/gemma-3-270m-math-edited/`
- Edited results: `results_edited/`
- Logs: gsm8k and math500 edited logs
- HF cache: model cached locally

## Next Steps (Recommendations)
1. **Try more targeted edits:** Head amplification is coarse. Try ROME/MEMIT style edits to MLP layers (causal tracing to find mediating layers for arithmetic). This is more compute-intensive but more targeted.
2. **Circuit analysis:** Use activation patching/head importance to identify specific heads/neurons critical for math, then amplify/prune more precisely.
3. **Task vectors:** If a compatible math-finetuned Gemma 3 270M variant exists, compute task vector `θ_math - θ_base` and try scaled addition (α search). Architecture must match.
4. **Iterative search:** With 2x T4s, can parallelize (analysis on one GPU, quick evals on other). Keep edits minimal and validate on small subsets before full benchmarks.
5. **Track carefully:** Save each edited checkpoint separately with notes about what was changed. Re-run both GSM8K and MATH-500 for fair comparison.

## Quick Reference Commands (VM)
```bash
# Check running benchmarks
pgrep -af lm_eval

# Check results
ls results_edited/

# Check logs
tail -30 logs/gsm8k_edited_*.log
tail -30 logs/math500_edited_*.log
```

## Notes
- Don't run full benchmarks on local laptop. Use CPU server or GPU VM.
- Head amplification approach tried first; results show minimal improvement. More sophisticated weight surgery likely needed for meaningful gains on MATH-500.
- All work done without finetuning/retraining as requested.