# RELEASE BUILD REPORT

## Build Target
- `${PROJECT_ROOT}/release/evbiox-lung-public`

## Created Directory Skeleton
- `docs/`
- `nsclc_swinunetr/scripts/`
- `nsclc_swinunetr/configs/`
- `datalist/`
- `scripts/`
- `demo/interactive/`
- `demo/assets/csv/`
- `demo/assets/stl/`
- `demo/assets/images/`
- `data/`

## Copied Files (Whitelist)
### Docs
- `docs/PROJECT_OVERVIEW.md`
- `docs/METHOD_PIPELINE.md`
- `docs/DEMO_ASSETS.md`
- `docs/SHOWCASE_STRUCTURE_PLAN.md`
- `docs/GITHUB_RELEASE_CHECKLIST_MINI.md`

### Core Scripts
- `nsclc_swinunetr/scripts/01_extract_series.py`
- `nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py`
- `nsclc_swinunetr/scripts/04_audit.py`
- `nsclc_swinunetr/scripts/05_make_datalist_nsclc.py`
- `nsclc_swinunetr/scripts/06_train_seg_nsclc.py`
- `nsclc_swinunetr/scripts/17_fusion_report_allin1_strict.py`

### Demo Assets
- `demo/interactive/interactive_report_v1.0.html`
- `demo/assets/csv/R01-056_tumor_metrics.csv`
- `demo/assets/csv/risk_groups.csv`
- `demo/assets/csv/km_curves_q3.csv`
- `demo/assets/stl/R01-056_tumor_gt.stl`
- `demo/assets/stl/R01-056_tumor_pred.stl`

## Generated/Completed Files
- `README.md`
- `.gitignore`
- `environment.yml`
- `scripts/run_demo_placeholder.md`
- `data/README.md`
- `demo/assets/images/README.md`
- `nsclc_swinunetr/configs/extract_nsclc.example.yaml`
- `nsclc_swinunetr/configs/preprocess_nsclc_vessel.example.yaml`
- `nsclc_swinunetr/configs/audit_roi.example.yaml`
- `nsclc_swinunetr/configs/datalist_nsclc.example.yaml`
- `nsclc_swinunetr/configs/train_seg_nsclc.example.yaml`
- `nsclc_swinunetr/configs/fusion_report_nsclc.example.yaml`

## Skipped by Policy
- `outputs/`
- `data/raw/`
- `data/processed/`
- `data/raw_data/`
- `.git/`
- `venv/`, `renv/`, `__pycache__/`, `cache/`, `logs/`, `tmp/`, `node_modules/`
- historical snapshots / duplicated export folders

## Conditional Skip
- `R01-056_lung.stl` was not copied due to size threshold in this build step.

## Sanitization Applied
- Replaced obvious `D:/...` / `D:\...` absolute path strings in copied text files with `${PROJECT_ROOT}` placeholder.
- Public configs provided as `.example.yaml` templates to avoid leaking local environment details.

## Manual Review TODO (Before Push)
- Verify copied scripts do not contain remaining hidden local/internal references.
- Decide whether to include `R01-056_lung.stl` based on final repo size budget.
- Add 3-5 screenshots into `demo/assets/images/`.
- Validate `environment.yml` dependency set in a clean environment.
- Run final `git status` whitelist check before initial commit.
