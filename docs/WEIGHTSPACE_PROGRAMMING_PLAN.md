# Weight-Space Programming Plan for Gemma 3 270M Math Improvement

## Baseline (from repo)
- **GSM8K (CoT), CPU server**: flexible exact_match = 0.0182 ± 0.0037 (1.82%), strict = 0.0114 ± 0.0029 (1.14%)
- **MATH-500, single T4**: exact_match = 0.002 ± 0.002 (0.2%)

## Goal
Improve scores on the same benchmarks via direct weight editing (no finetuning/retraining). Edit weights, save modified checkpoint, benchmark later on appropriate hardware.

## Compute Requirements
- **Editing**: With 2x T4 GPUs now available, causal tracing/activation patching will be much faster than CPU. 270M model fits easily on a single T4 (float32 ~1GB, float16 even less). Can do all editing on the GPU VM.
- **Benchmarking**: Run on the same GPU VM (or keep using it). Matches previous MATH-500 setup (single T4).

## Weight-Space Programming Techniques (No Finetuning/Retraining)

### 1) Attention Head Amplification (easiest, lowest risk)
- **Method**: Identify important heads for math via activation patching/head importance; scale attention output projection weights for those heads by factor α > 1 (e.g., 1.1-2.0).
- **Targets**: Gemma 3 270M has 4 heads/layer, 18 layers - easy to sweep candidates.
- **Rationale**: Pure weight scaling, reversible, no gradient updates.

### 2) ROME-style MLP Fact Edits (target arithmetic facts)
- **Method**: Use causal tracing to locate mediating MLP layers for arithmetic facts (e.g., "2+3=", multiplication tables). Apply rank-1 update to MLP weight matrices.
- **Targets**: Middle layers (likely ~6-12) based on small-model literature.
- **Rationale**: Helps with direct fact recall; may improve both GSM8K and MATH-500 on basic operations.

### 3) Task Vector Addition (if available)
- **Method**: If a compatible math task vector τ = θ_math_ft - θ_base exists for same architecture family, apply θ' = θ_base + α*τ.
- **Status**: No math-finetuned Gemma 3 270M in repo; would need external task vector. Not currently available.

### 4) Function Vector Weight Injection (experimental)
- **Method**: Extract function vector from activations mediating math reasoning, inject into weights (approximate). More experimental than above.

## Implementation Plan
1. **Setup environment on GPU VM** (2x T4): Install transformers, torch, lm-eval, accelerate. Load model.
2. **Causal analysis**: Run activation patching/head importance on small math examples to identify targets (much faster on GPUs).
3. **Apply edits**: Start with head amplification, then add targeted MLP fact edits (ROME-style).
4. **Save checkpoint**: Save modified model to `models/gemma-3-270m-math-edited/` (safetensors).
5. **Benchmark**: Run lm-eval with same configs as baselines (gsm8k_cot, hendrycks_math500) on the GPU VM; record results.
6. **Iterate**: If improvements seen, try additional edits; if side effects appear, revert/scalar down.

## Notes
- All edits are weight manipulations only - no finetuning, no gradient descent over benchmark datasets.
- Keep edits minimal and reversible (save original weights/checkpoints).
- Compare against baselines using same evaluation setup (shot count, prompts, generation settings).
