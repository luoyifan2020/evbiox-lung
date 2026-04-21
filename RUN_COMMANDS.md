# RUN COMMANDS

## 1) `nsclc_swinunetr/scripts/01_extract_series.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/01_extract_series.py --cfg nsclc_swinunetr/configs/extract_nsclc.example.yaml`
  - 或：`python nsclc_swinunetr/scripts/01_extract_series.py --raw_dicom_root "${PROJECT_ROOT}/data/raw/dicom/nsclc" --raw_nifti_root "${PROJECT_ROOT}/data/raw/nifti/nsclc"`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：`--cfg` / `--raw_dicom_root` / `--raw_nifti_root`
- **关键输出**：NIfTI 与 `meta.json` 到 `raw_nifti_root`

## 2) `nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py --cfg nsclc_swinunetr/configs/preprocess_nsclc_vessel.example.yaml`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：`--cfg`、`--num_workers`、`--log-level`
- **关键输出**：预处理结果、血管/PA 相关掩膜、QC 图与统计（由 cfg 指定输出目录）

## 3) `nsclc_swinunetr/scripts/04_audit.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/04_audit.py summary --cfg nsclc_swinunetr/configs/audit_roi.example.yaml`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：子命令 `summary` + `--cfg`（可覆盖 `--roi_root`、`--cube_size`、`--num_workers`）
- **关键输出**：审计 CSV（如 `audit_roi.csv`）

## 4) `nsclc_swinunetr/scripts/05_make_datalist_nsclc.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/05_make_datalist_nsclc.py --cfg nsclc_swinunetr/configs/datalist_nsclc.example.yaml`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：`--cfg`（必填）
- **关键输出**：`datalist_nsclc_fold*.json`、`datalist_nsclc_kfold.json`、`datalist_nsclc_test.json`、`patient_folds.csv`、`roi_manifest.csv`

## 5) `nsclc_swinunetr/scripts/06_train_seg_nsclc.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/06_train_seg_nsclc.py --cfg nsclc_swinunetr/configs/train_seg_nsclc.example.yaml`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：`--cfg`（必填）
- **关键输出**：`best.pth`、`last.pth`、`metrics.csv`（输出目录由 cfg 的 `out_dir` 指定）

## 6) `nsclc_swinunetr/scripts/17_fusion_report_allin1_strict.py`
- **最小可运行命令**
  - `python nsclc_swinunetr/scripts/17_fusion_report_allin1_strict.py --cfg nsclc_swinunetr/configs/fusion_report_nsclc.example.yaml`
- **是否支持 --config**：否（支持 `--cfg`）
- **关键输入参数**：`--cfg`（必填）、`--verbose`（可选）
- **关键输出**：`metrics.json`、`oof.csv` 与报告图表（输出目录由 cfg 的 `out_dir` 指定）

---

## README 适合写的 Quick Start 命令
```bash
python nsclc_swinunetr/scripts/01_extract_series.py --cfg nsclc_swinunetr/configs/extract_nsclc.example.yaml
python nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py --cfg nsclc_swinunetr/configs/preprocess_nsclc_vessel.example.yaml
python nsclc_swinunetr/scripts/04_audit.py summary --cfg nsclc_swinunetr/configs/audit_roi.example.yaml
python nsclc_swinunetr/scripts/05_make_datalist_nsclc.py --cfg nsclc_swinunetr/configs/datalist_nsclc.example.yaml
python nsclc_swinunetr/scripts/06_train_seg_nsclc.py --cfg nsclc_swinunetr/configs/train_seg_nsclc.example.yaml
python nsclc_swinunetr/scripts/17_fusion_report_allin1_strict.py --cfg nsclc_swinunetr/configs/fusion_report_nsclc.example.yaml
```
