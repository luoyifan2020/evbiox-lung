# DEMO ASSETS

## Recommended Public Demo Inputs
Reference source: `EVBioX_Demo_Root` (curated and copied, not linked by absolute path).

### High-priority assets
- `interactive_report_v1.0.html`
- `R01-056_tumor_metrics.csv`
- `risk_groups.csv`
- `km_curves_q3.csv`
- `R01-056_tumor_gt.stl`
- `R01-056_tumor_pred.stl`

### Optional asset
- `R01-056_lung.stl` (larger file; include only if repo size permits)

## Suggested In-Repo Placement
- `demo/interactive/interactive_report_v1.0.html`
- `demo/assets/csv/*.csv`
- `demo/assets/stl/*.stl`

## Curation Rules
- Remove patient identifiers and private metadata before publishing.
- Avoid absolute local paths and internal system references.
- Keep only a minimal, representative sample for showcase.
- Do not include raw DICOM or full training outputs.

## Display Focus
- What the workflow produces (visual and tabular outputs)
- How to interpret demo artifacts at a high level
- Reproducible method path (not exhaustive internal history)
