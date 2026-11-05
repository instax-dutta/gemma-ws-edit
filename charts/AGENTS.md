# DOX: charts

## Purpose
Chart data (CSV) and generation script for gemma-ws-edit visualizations.

## Ownership
Visualization assets and source data.

## Local Contracts
- Store source data as CSV in charts/data/.
- Generate charts via charts/generate_charts.py.
- Output PNG and SVG to assets/charts/.
- Do not commit large binary assets unless needed for display.

## Work Guidance
- Regenerate charts whenever result data changes.
- Keep chart styling consistent (see PALETTE in generate_charts.py).
- Use publication-quality settings (150 DPI, tight bounding boxes).

## Verification
- Chart script runs without errors.
- Output PNG/SVG files exist in assets/charts/.

## Child DOX Index
None.