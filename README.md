# gemma-ws-edit

Best checkpoint on Hugging Face: [saidutta69/gemma-3-270m-math-rome-best](https://huggingface.co/saidutta69/gemma-3-270m-math-rome-best) with tables and figures. Code and logs are in this repo under instax-dutta/gemma-ws-edit. Weights live on Hugging Face.

Direct weight editing for Gemma 3 270M math, no finetuning and no retraining.

## What this project does

gemma-ws-edit tests one idea: change the weights directly and see if math gets better, without fine-tuning.

We use Google's Gemma 3 270M and test on GSM8K and MATH-500. Instead of gradient descent over many examples, we scale attention heads, scale MLP layers ROME-style, or add a task vector. Each change is small and reversible. We then rerun the same benchmarks and compare to a fixed baseline.

## Why this matters

Fine-tuning a 270M model on math takes compute, time, and labeled data. Weight editing asks if you can get better reasoning from a model you already have by changing its weights by hand. Even a small, clear gain would suggest a light way to improve a model without paying for retraining.

## What we tried and what happened

MPS-only comparison. Same setup for each run: float32, batch 4, same lm-eval tasks. T4 numbers below are old reference only.

| Approach | GSM8K flexible | GSM8K strict | MATH-500 | Correct /1319 |
|---|---:|---:|---:|---:|
| Baseline MPS | 1.82% | 1.14% | 0.20% | 24 |
| v2 Head all | 1.82% | 1.14% | 0.20% | 24 |
| v3 MLP mid 6-12 alpha 1.5 (best published) | 2.05% | 1.36% | 0.20% | 27 |
| v4 Head plus MLP | 2.05% | 1.36% | 0.20% | 27 |
| v5 MLP alpha 2.0 | 2.05% | 1.36% | 0.20% | 27 |
| v6 MLP alpha 2.5 | 2.05% | 1.29% | 0.20% | 27 |
| v7 MLP late 9-17 | 1.59% | 1.14% | 0.20% | 21 |
| v8 ROME rank-1 | 1.59% | 1.14% | 0% | 21 |
| v9 Task vector 0.2 | 1.82% | 1.44% | 0% | 24 |

### Baseline MPS
- GSM8K CoT: exact_match flexible 0.0182 plus or minus 0.0037 (24/1319), strict 0.0114 plus or minus 0.0029 (15/1319)
- MATH-500: 0.002 plus or minus 0.002 (1/500, 0-shot)
- Full JSONs in `results/full_baseline_local/`

### Best published
- `saidutta69/gemma-3-270m-math-rome-best` is v3, MLP layers 6-12 alpha 1.5 (down_proj 640x2048, 7 layers, 9.17M params, 3.4% changed)
- Gain is +3 correct (27 vs 24), below the bar of 34 (baseline + 2 times stderr). MATH-500 did not change.

Full sweep figures are in `assets/charts/`, including `mps_sweep_all.png` for all 9 runs.

## Results visualization

### Baseline vs weight-edited

![Baseline vs Edited](assets/charts/baseline_vs_edited.png)

![MPS Sweep All Runs](assets/charts/mps_sweep_all.png)

### MATH-500 by subject

![MATH-500 by Subject](assets/charts/math500_by_subject.png)

### MATH-500 by difficulty level

![MATH-500 by Level](assets/charts/math500_by_level.png)

### Summary dashboard

![Summary Dashboard](assets/charts/summary_dashboard.png)

Charts come from data in `charts/data/` and the script `charts/generate_charts.py`. PNG and SVG copies are in `assets/charts/`.

## Techniques we explored

1. Attention head amplification. Scale output projection weights of attention heads. Low risk and reversible.
2. ROME-style MLP edits. Use causal tracing to find MLP layers for arithmetic facts, then apply rank-1 updates to those weights.
3. Task vector addition. If a compatible math-tuned Gemma 3 270M exists, compute theta_math minus theta_base and add a scaled copy to the base weights.
4. Function vector injection. Take function vectors from activations that drive math reasoning and inject them into weights with an approximation.

All edits change weights only. No fine-tuning and no gradient descent on benchmark data. We keep each edit small and compare against the same baseline with the same eval setup.

## Repo structure

```
gemma-ws-edit/
├── README.md           # This file, overview, results, quickstart
├── AGENTS.md           # Root project context for agents and contributors
├── GATES.md            # Checklist of documentation and plan gates
├── LICENSE             # MIT license
├── .gitignore          # Python and model cache ignores
├── scripts/            # Benchmark execution scripts
│   ├── run_bench.py    # Runs lm-eval benchmarks (GSM8K, MATH-500)
│   ├── quick_eval.py   # Quick local sanity check on a few problems
│   └── AGENTS.md       # Scripts directory context
├── charts/            # Chart data (CSV) and generation script
│   ├── data/          # CSV data files for charts
│   ├── generate_charts.py  # Matplotlib chart generator
│   └── AGENTS.md      # Charts directory context
├── assets/            # Generated visual assets
│   └── charts/        # PNG and SVG charts (embedded in README)
├── results/           # Evaluation outputs (JSON, JSONL samples)
│   ├── google__gemma-3-270m/   # Baseline results
│   └── AGENTS.md       # Results directory context
├── logs/               # Runtime execution logs
│   └── AGENTS.md       # Logs directory context
├── models/             # Model artifacts and checkpoints (not committed)
│   └── AGENTS.md       # Models directory context
├── docs/               # Research notes and detailed documentation
│   ├── WEIGHTSPACE_PROGRAMMING_PLAN.md   # Full editing plan and techniques
│   ├── BENCHMARK_LOG.md                  # Complete benchmark run log
│   ├── HANDOFF.md                          # Handoff notes for GPU VM work
│   └── AGENTS.md       # Docs directory context
└── dox/                # DOX tree documentation
    └── AGENTS.md       # DOX directory context
```

## Quick start

### Prerequisites

- Python 3.10+
- `torch`, `transformers`, `lm-eval`, `accelerate`

### Run a benchmark

```bash
# Clone the repo
git clone https://github.com/<your-username>/gemma-ws-edit.git
cd gemma-ws-edit

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install lm-eval transformers torch accelerate

# Run GSM8K + MATH-500 on the base model
python scripts/run_bench.py
```

The script downloads `google/gemma-3-270m` from HuggingFace, runs the benchmarks, and writes results to `results/` and logs to `logs/`.

### Quick local test

```bash
python scripts/quick_eval.py
```

This runs the model on a few GSM8K-style word problems and prints the generations. It is a quick way to check the setup before a full run.

## Compute notes

- Laptop (Mac MPS): slow. Fine for quick tests, not full benchmarks.
- CPU server: workable for GSM8K, very slow per sample.
- GPU VM (2x Tesla T4): best for editing and benchmarking. A 270M model loads in about 1GB float32. Tracing and benchmarks run much faster there.

Run one heavy job at a time on shared CPU servers.

## Contributing

This is a research project. Contributions are welcome.

- Try a new weight editing technique (ROME, MEMIT, task vectors, circuit pruning)
- Improve the benchmark scripts or add new tasks (Minerva, MATH, etc.)
- Reproduce or extend the baseline results on different hardware
- Document failed experiments (negative results matter too)

Open an issue to discuss, or submit a pull request.

## Citation

If you use this work, please cite:

```
@misc{gemma-ws-edit,
  title = {gemma-ws-edit: Direct Weight-Space Editing for Gemma 3 270M Math Reasoning},
  author = {gemma-ws-edit contributors},
  year = {2026},
  url = {https://github.com/<your-username>/gemma-ws-edit}
}
```

## License

MIT License. See LICENSE for details.
