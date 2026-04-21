# METHOD PIPELINE

## One-Line Method
A staged pipeline combining CT segmentation and multimodal risk fusion for NSCLC analysis.

## Pipeline Steps
1. **Data extraction**
   - Script: `nsclc_swinunetr/scripts/01_extract_series.py`
   - Purpose: convert/select CT/SEG series into standardized NIfTI assets.

2. **Preprocessing and vessel processing**
   - Script: `nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py`
   - Purpose: preprocess CT, generate/reuse lung-vessel or PA-related masks, produce QC outputs.

3. **ROI/data audit (optional but recommended)**
   - Script: `nsclc_swinunetr/scripts/04_audit.py`
   - Purpose: basic quality checks for ROI completeness and mask integrity.

4. **Datalist generation**
   - Script: `nsclc_swinunetr/scripts/05_make_datalist_nsclc.py`
   - Purpose: patient-level stratified splits and MONAI-compatible datalist files.

5. **Segmentation training**
   - Script: `nsclc_swinunetr/scripts/06_train_seg_nsclc.py`
   - Purpose: train SwinUNETR segmentation model and save checkpoints/metrics.

6. **Fusion report and survival/risk visualization**
   - Script: `nsclc_swinunetr/scripts/17_fusion_report_allin1_strict.py`
   - Purpose: aggregate OOF/test risk outputs and generate report figures/tables.

## What This Repo Intentionally Does Not Include
- Full raw medical datasets
- Full historical experiment outputs
- Claims of benchmark results without external validation context
