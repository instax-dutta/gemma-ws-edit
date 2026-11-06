# Handoff: Gemma 3 270M weight-space programming for math

## Project overview
- Goal: improve math for `google/gemma-3-270m` by direct weight edits, with no finetuning or retraining. Test on GSM8K and MATH-500.
- Repo: gemma-ws-edit (DOX tree initialized). Baselines are in `AGENTS.md` and `docs/BENCHMARK_LOG.md`.

## Baselines
- GSM8K (CoT, CPU server, float32, lm-eval): flexible 0.0182 ± 0.0037 (1.82%), strict 0.0114 ± 0.0029 (1.14%)
- MATH-500 (T4, float32, lm-eval): 0.002 ± 0.002 (0.2%), 500 samples, 0-shot
- MPS baseline (MacBook Air M2, 8GB) matched the same numbers. Later runs use MPS vs MPS.

## Compute environment
- Original benchmarks: moved from laptop (MPS slow at first) to CPU server (12 vCPUs, 47GB RAM, no GPU). One heavy job at a time.
- Weight editing and benchmarks: 2x Tesla T4 GPU VM. Enough for the 270M model (about 1GB) and tracing or benchmarks.
- Later MPS runs: full GSM8K + MATH-500 finished in 44m54s on M2. Recommendation stands: weight editing needs 8-16GB RAM. Causal tracing is slow on CPU.

## What was done
1. Planning: created `docs/WEIGHTSPACE_PROGRAMMING_PLAN.md` with head amplification, ROME/MEMIT for MLPs, and task vectors. Updated `AGENTS.md` goal. Created `GATES.md`.
2. First weight edit: on the GPU VM, copied the repo and HF model. Applied attention head amplification (scale attention output projections). Saved the checkpoint to `models/gemma-3-270m-math-edited/`.
3. Benchmarked the edited model: ran GSM8K and MATH-500 with lm-eval.

## Results (edited vs baseline, first pass)
- GSM8K: flexible 0.0197 ± 0.0038 (+0.15pp), strict 0.0121 ± 0.0030 (+0.07pp)
- MATH-500: 0.002 ± 0.002 (no change)

Head amplification gave a minimal gain on GSM8K and none on MATH-500. Later MPS sweeps (v2-v9) show the same plateau at 27/1319. See `charts/data/mps_all_runs.csv` and the published checkpoint `saidutta69/gemma-3-270m-math-rome-best`.

## Files and locations
### In repo (local)
- `AGENTS.md` - updated goals and baselines
- `docs/BENCHMARK_LOG.md` - full log with baselines and edited results
- `docs/WEIGHTSPACE_PROGRAMMING_PLAN.md` - detailed plan and approaches
- `GATES.md` - unlazy gates
- `scripts/run_bench.py`, `scripts/quick_eval.py`, `scripts/edit_weights.py`, `scripts/run_full_bench.py` - benchmark and edit scripts
- `results/full_baseline_local/` - MPS baseline results

### On GPU VM (first pass, now reference)
- Repo copy and model cache
- Edited model: `models/gemma-3-270m-math-edited/`
- Edited results: `results_edited/`
- Logs: gsm8k and math500 edited logs
- HF cache: model cached locally

## Next steps (from first pass, now mostly done)
1. More targeted edits: head amplification is coarse. Try ROME/MEMIT edits to MLP layers with causal tracing for arithmetic. More compute-intensive but more targeted.
2. Circuit analysis: use activation patching or head importance to find heads or neurons for math, then amplify or prune with more care.
3. Task vectors: if a compatible math-tuned Gemma 3 270M exists, compute `theta_math - theta_base` and try scaled addition (alpha search). Architecture must match. Done for Nirav-Madhani candidate at alpha 0.2, MATH fell to 0.
4. Iterative search: keep edits minimal and validate on small subsets before full benchmarks.
5. Track carefully: save each edited checkpoint separately with notes. Rerun GSM8K and MATH-500 for fair comparison.

## Quick reference commands (VM)
```bash
# Check running benchmarks
pgrep -af lm_eval

# Check results
ls results/full_baseline_local/

# Check logs
tail -30 logs/full_baseline_local.out
```

## Notes
- Do not run full benchmarks on a slow local setup without a plan. Use MPS batch 4 or a GPU VM.
- Head amplification was tried first because it is fast. Results show a minimal gain. More targeted weight surgery is needed for meaningful MATH-500 gains. Our sweeps did not find one for 270M.
- All work was done without finetuning or retraining.
