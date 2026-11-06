# Weight-space programming plan for Gemma 3 270M math

## Baseline
- GSM8K (CoT), CPU server: flexible 0.0182 ± 0.0037 (1.82%), strict 0.0114 ± 0.0029 (1.14%)
- MATH-500, single T4: 0.002 ± 0.002 (0.2%)
- MPS baseline (MacBook Air M2, 8GB) reproduced the same numbers with the same lm-eval tasks. All later comparisons use MPS vs MPS.

## Goal
Improve scores on the same benchmarks by editing weights directly, with no finetuning or retraining. Edit weights, save the new checkpoint, and rerun the benchmarks on the same setup.

## Compute
- Editing: causal tracing and activation patching are much faster on GPU than on CPU. The 270M model fits in about 1GB float32, so it runs in RAM on a laptop and on a single T4.
- Benchmarking: MPS took 44m54s for full GSM8K + MATH-500 here. The T4 VM took 3193s. MPS was faster because of unified memory and batch 4.

## Techniques (no finetuning/retraining)

### 1. Attention head amplification (easiest, low risk)
- Method: find important heads for math with activation patching or head importance, then scale attention output projection weights for those heads by alpha > 1 (1.1 to 2.0).
- Targets: Gemma 3 270M has 4 heads per layer and 18 layers, so a sweep is small.
- Why it is useful: pure weight scaling, reversible, no gradient updates.

### 2. ROME-style MLP edits (arithmetic facts)
- Method: use causal tracing to find MLP layers for arithmetic facts (for example "2+3=" and multiplication tables), then apply a rank-1 update to those MLP weights.
- Targets: middle layers, about 6-12, following small-model reports.
- Why it is useful: helps direct fact recall. It may improve GSM8K and MATH-500 on basic operations.

### 3. Task vector addition (if available)
- Method: if a compatible math-tuned model exists for the same architecture, set theta' = theta_base + alpha * (theta_math - theta_base).
- Status: We found public candidates and verified shape match for one (Nirav-Madhani/gemma3-270m-grpo-math, 236 tensors). At alpha 0.2 the strict score rose but MATH-500 fell to 0.

### 4. Function vector injection (experimental)
- Method: take a function vector from activations that drive math reasoning and inject it into weights with an approximation. More experimental than the three above.

## Implementation plan
1. Set up the environment (transformers, torch, lm-eval, accelerate) and load the model.
2. Causal analysis: run activation patching or head importance on small math examples to find targets.
3. Apply edits: start with head amplification, then targeted MLP fact edits (ROME-style).
4. Save checkpoint: save the edited model to `models/gemma-3-270m-math-edited/` in safetensors.
5. Benchmark: run lm-eval with the same configs as the baselines (`gsm8k_cot`, `hendrycks_math500`) and record results.
6. Iterate: if scores improve, try more edits. If side effects appear, revert or scale down.

## Notes
- All edits are weight changes only. No finetuning and no gradient descent over benchmark data.
- Keep edits minimal and reversible (save original weights and checkpoints).
- Compare against baselines with the same eval setup (shot count, prompts, generation settings).
