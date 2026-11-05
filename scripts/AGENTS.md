# DOX: scripts

## Purpose
Benchmark execution scripts for Gemma 3 270M math evaluation and weight-space editing experiments.

## Ownership
Local benchmark runner scripts.

## Local Contracts
- Scripts must be reproducible and log all parameters.
- Store outputs to ../results and ../logs.
- Document dependencies and hardware used.

## Work Guidance
- Keep scripts idempotent where possible.
- One heavy benchmark job at a time on shared CPU servers.
- Use GPU VM for weight editing and full benchmark runs.

## Verification
- Scripts should produce valid JSON outputs.
- Check logs for errors before concluding runs.

## Child DOX Index
None.