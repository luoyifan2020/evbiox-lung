# PROJECT OVERVIEW

## Project Role
`evbiox-lung` is the core development workspace for NSCLC segmentation + multimodal risk analysis.

## Recommended Public Showcase Structure
- `README.md`
- `.gitignore`
- `docs/`
  - `PROJECT_OVERVIEW.md`
  - `METHOD_PIPELINE.md`
  - `DEMO_ASSETS.md`
- `nsclc_swinunetr/`
  - `scripts/` (core steps)
  - `configs/` (run templates)
- `datalist/` (lightweight metadata only)
- `scripts/`
  - `run_demo_placeholder.md`

## Core vs Legacy
- Core for public showcase:
  - `nsclc_swinunetr/scripts/{01,02,05,06,17}_*.py`
  - related config files in `nsclc_swinunetr/configs/`
  - curated docs and demo notes
- Legacy/heavy content (exclude from public by default):
  - `outputs/`
  - `data/raw/`, `data/processed/`, `data/raw_data/`
  - duplicated test/copy scripts unless needed for demonstration

## Publishing Goal
Present a clear end-to-end workflow and representative visual artifacts without exposing raw data, private paths, or internal experiment clutter.
