# DOX: logs

## Purpose
Execution logs from benchmark runs and weight-editing experiments.

## Ownership
Runtime logs.

## Local Contracts
- Capture stdout/stderr from all runs.
- Include timestamps and configuration.
- Do not commit sensitive information.

## Work Guidance
- Rotate or archive old logs if needed.
- Use grep for progress counters (e.g., "d/d total") rather than tailing truncated output.

## Verification
- Check for errors after runs.
- Confirm process completion before drawing conclusions.

## Child DOX Index
None.