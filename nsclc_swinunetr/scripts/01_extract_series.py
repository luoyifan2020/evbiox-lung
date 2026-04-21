#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_extract_series_standalone.py — 2025-09-10
从 raw_dicom_root 抽取/转换 CT 与分割(SEG)，输出到 raw_nifti_root，并写 meta。

特点
- 完全独立：不依赖 nsclc_swinunetr.utils 包内导入
- 兼容两种用法：直接传参 或 使用 --cfg 读取 YAML（含 extract 段）
- 逻辑参考你现有 01_extract_series.py + dicom_io.py（扫描 Series、胸部 CT 过滤、SEG 关联）并去依赖化

依赖
  pip install SimpleITK pydicom pyyaml

示例（PowerShell）：
  conda activate evbiox-gpu
  python nsclc_swinunetr/scripts/01_extract_series_standalone.py `
    --raw_dicom_root "${PROJECT_ROOT}/data/raw/dicom/nsclc" `
    --raw_nifti_root "${PROJECT_ROOT}/nsclc_swinunetr/outputs/raw_nifti_root" `
    --log-level info

或用配置：
  python nsclc_swinunetr/scripts/01_extract_series_standalone.py --cfg configs/extract_nsclc.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

import pydicom
import SimpleITK as sitk
import yaml

# ------------------------------ 日志 ------------------------------ #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ========================= 内嵌版 dicom_io ========================= #
# 参考你的 utils/dicom_io.py，做了少量防御性处理（异常吞吐、健壮性）以便独立使用。  # noqa

_CHEST_KEY = ("chest", "thorax", "lung")
_EXCLUDE_KEY = ("pet", "scout", "localizer", "mip", "dose")

def dcm_header(path: Path) -> pydicom.Dataset:
    """懒加载 DICOM header；无法解析时抛异常由上游捕获。"""
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

def walk_series(case_dir: Path):
    """扫描病例目录下所有 .dcm，按 SeriesInstanceUID 分组。"""
    series_dict: dict[str, list[Path]] = {}
    for dcm in case_dir.rglob("*.dcm"):
        try:
            ds = dcm_header(dcm)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid:
                continue
        except Exception:
            continue
        series_dict.setdefault(uid, []).append(dcm)
    for uid, files in series_dict.items():
        yield uid, sorted(files)

def series_meta(files: List[Path]) -> dict:
    """返回一个 Series 的关键信息字典，用于写 meta.json。"""
    ds = dcm_header(files[0])
    return {
        "series_uid": getattr(ds, "SeriesInstanceUID", ""),
        "study_uid": getattr(ds, "StudyInstanceUID", ""),
        "series_desc": getattr(ds, "SeriesDescription", ""),
        "modality": getattr(ds, "Modality", ""),
        "body_part": getattr(ds, "BodyPartExamined", ""),
        "manufacturer": getattr(ds, "Manufacturer", ""),
        "model_name": getattr(ds, "ManufacturerModelName", ""),
        "kvp": float(getattr(ds, "KVP", 0) or 0),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0) or 0),
        "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [0, 0])],
        "num_slices": len(files),
        "first_dcm": files[0].name,
    }

def chest_ct_series(case_dir: Path):
    """
    只返回符合“胸部 CT”特征的 Series（Modality=CT 且
    SeriesDescription/BodyPartExamined 含 chest|thorax|lung）。
    """
    for uid, files in walk_series(case_dir):
        try:
            ds = dcm_header(files[0])
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        desc = (getattr(ds, "SeriesDescription", "") or "").lower()
        body = (getattr(ds, "BodyPartExamined", "") or "").lower()
        if any(k in desc or k in body for k in _CHEST_KEY) and not any(b in desc for b in _EXCLUDE_KEY):
            yield files

def pick_ct_series(case_dir: Path) -> Optional[Path]:
    """
    fallback：选择“切片数最多”的 CT，返回其第一张切片路径。
    若不存在 CT 则返回 None。
    """
    best: Optional[Tuple[Path, int]] = None
    for uid, files in walk_series(case_dir):
        try:
            ds = dcm_header(files[0])
            if getattr(ds, "Modality", "") != "CT":
                continue
        except Exception:
            continue
        if best is None or len(files) > best[1]:
            best = (files[0], len(files))
    return best[0] if best else None

# 允许动态修改的 SEG 文件名关键词
SEG_NAME_HINT: Sequence[str] = ("segmentation", "result", "seg")

def iter_seg_files(case_dir: Path) -> Generator[Path, None, None]:
    """遍历所有可能的分割文件（NIfTI 或 DICOM-SEG）。"""
    for fp in case_dir.rglob("*"):
        if not fp.is_file():
            continue
        low = fp.name.lower()
        # NIfTI：基于关键词过滤
        if low.endswith((".nii.gz", ".nii")) and any(h in low for h in SEG_NAME_HINT):
            yield fp
        # DICOM-SEG：看 Modality
        elif fp.suffix.lower() == ".dcm":
            try:
                if dcm_header(fp).Modality.upper() == "SEG":
                    yield fp
            except Exception:
                continue

def pick_seg_for_ct(case_dir: Path, ct_series_uid: str | None = None) -> Optional[Path]:
    """
    精选一个最合适的分割文件，优先级：
    1. 若指定 ct_series_uid，返回“引用该 UID 的 DICOM-SEG”
    2. 第一个 NIfTI SEG
    3. 任何 DICOM-SEG
    """
    # 1) 精确引用
    if ct_series_uid:
        for fp in iter_seg_files(case_dir):
            if fp.suffix.lower() != ".dcm":
                continue
            try:
                ds = dcm_header(fp)
                for ref in getattr(ds, "ReferencedSeriesSequence", []):
                    if getattr(ref, "SeriesInstanceUID", "") == ct_series_uid:
                        return fp
            except Exception:
                continue
    # 2) NIfTI
    for fp in iter_seg_files(case_dir):
        if fp.suffix.lower() in (".nii", ".nii.gz"):
            return fp
    # 3) 任何 DICOM-SEG
    for fp in iter_seg_files(case_dir):
        return fp
    return None

# ========================= 工具函数（本脚本） ========================= #

def dicom_to_nii(dcm_first_slice: Path, out_path: Path) -> None:
    """把整个 DICOM Series 转为单个 NIfTI (.nii.gz)。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesFileNames(str(dcm_first_slice.parent))
    reader.SetFileNames(series)
    img = reader.Execute()
    sitk.WriteImage(img, str(out_path))

def dicom_seg_to_nii(seg_path: Path, out_path: Path) -> bool:
    """
    尝试把 DICOM-SEG 转成 NIfTI。成功返回 True，否则 False（并记录 warning）。
    说明：这一步在某些数据上可能需要更专业的 DICOM-SEG 解析器；先用 SimpleITK 尝试。
    """
    try:
        img = sitk.ReadImage(str(seg_path))
        sitk.WriteImage(img, str(out_path))
        return True
    except Exception as e:  # pragma: no cover
        logging.warning(f"SimpleITK 转 SEG 失败：{e}")
        return False

def copy_or_convert_seg(seg_src: Path, dst_nii: Path, save_dcm_fallback: bool = False) -> None:
    """根据 seg_src 类型（NIfTI / DICOM-SEG）完成复制或转换。"""
    dst_nii.parent.mkdir(parents=True, exist_ok=True)
    low = seg_src.suffix.lower()
    if low in (".nii", ".gz"):  # 既支持 .nii 也支持 .nii.gz
        shutil.copy2(seg_src, dst_nii)
        return
    # DICOM-SEG
    if dicom_seg_to_nii(seg_src, dst_nii):
        return
    # 转换失败
    if save_seg_as_dcm:
        shutil.copy2(seg_src, dst_nii.with_suffix(".dcm"))
    else:
        raise RuntimeError(f"Unable to convert SEG: {seg_src}")

def write_meta(out_dir: Path, ct_files: List[Path], seg_src: Path | None, choose_reason: str) -> None:
    """生成 meta.json 和 slice_meta.csv。"""
    meta_ct = series_meta(ct_files)
    meta_seg = {
        "seg_file": seg_src.name if seg_src else None,
        "seg_modality": ("DICOM-SEG" if (seg_src and seg_src.suffix.lower() == ".dcm") else ("NIfTI" if seg_src else None)),
        "choose_reason": choose_reason,
    }
    if seg_src and seg_src.suffix.lower() == ".dcm":
        try:
            ds = dcm_header(seg_src)
            refs = [rs.SeriesInstanceUID for rs in getattr(ds, "ReferencedSeriesSequence", [])]
        except Exception:
            refs = []
        meta_seg["seg_referenced_series_uids"] = refs

    meta = {
        "tool_version": "evbiox-extract-standalone 2025-09-10",
        **meta_ct,
        **meta_seg,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # slice_meta.csv
    rows = []
    for fp in ct_files:
        try:
            ds = dcm_header(fp)
            sop = getattr(ds, "SOPInstanceUID", "")
            inst = getattr(ds, "InstanceNumber", "")
            z = getattr(ds, "ImagePositionPatient", [None, None, None])
            z = z[2] if isinstance(z, (list, tuple)) and len(z) >= 3 else ""
        except Exception:
            sop, inst, z = "", "", ""
        rows.append([fp.name, sop, inst, z])

    with open(out_dir / "slice_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "SOPUID", "InstanceNumber", "Z"])
        w.writerows(rows)

# ============================== 主流程 ============================== #

def load_cfg(cfg_path: Optional[str]) -> dict | None:
    if not cfg_path:
        return None
    return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

def process_case(case_dir: Path, out_root: Path, seg_name_hint: Sequence[str], save_seg_as_dcm: bool) -> None:
    """处理单个病例目录。"""
    start_t = time.perf_counter()
    case_id = case_dir.name
    out_dir = out_root / case_id

    # ---------- 1. 搜索 SEG（允许动态提示词） ----------
    global SEG_NAME_HINT
    SEG_NAME_HINT = tuple(seg_name_hint)

    seg_src = pick_seg_for_ct(case_dir, None)

    # ---------- 2. 选 CT ----------
    ct_files: List[Path] | None = None
    choose_reason = ""
    ref_uids = []
    if seg_src and seg_src.suffix.lower() == ".dcm":
        try:
            ds = dcm_header(seg_src)
            ref_uids = [rs.SeriesInstanceUID for rs in getattr(ds, "ReferencedSeriesSequence", [])]
        except Exception:
            ref_uids = []

    # 2-A SEG 精确引用
    if ref_uids:
        for uid, files in walk_series(case_dir):
            if uid in ref_uids:
                ct_files = files
                choose_reason = "by_seg_reference"
                break

    # 2-B 过滤胸部 CT
    if ct_files is None:
        chest = list(chest_ct_series(case_dir))
        if chest:
            ct_files = max(chest, key=len)
            choose_reason = "by_chest_filter"

    # 2-C fallback：切片数最多
    if ct_files is None:
        ct_first = pick_ct_series(case_dir)
        if not ct_first:
            logging.warning(f"[SKIP] {case_id}: 找不到 CT Series")
            return
        ct_files = [ct_first]
        choose_reason = "largest_ct_fallback"

    ct_first = ct_files[0]
    try:
        ct_series_uid = dcm_header(ct_first).SeriesInstanceUID
    except Exception:
        ct_series_uid = None

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 3. 转 / 复制 CT ----------
    ct_out = out_dir / "ct.nii.gz"
    if not ct_out.exists():
        logging.info(f"{case_id}: 转换 CT → NIfTI")
        dicom_to_nii(ct_first, ct_out)
    else:
        logging.debug(f"{case_id}: 已存在 ct.nii.gz，跳过转换")

    # ---------- 4. 选 / 转 SEG ----------
    if seg_src is None:
        seg_src = pick_seg_for_ct(case_dir, ct_series_uid)

    if seg_src:
        seg_out = out_dir / "seg.nii.gz"
        if not seg_out.exists():
            logging.info(f"{case_id}: 复制 / 转换分割 {seg_src.name}")
            try:
                copy_or_convert_seg(seg_src, seg_out, save_seg_as_dcm)
            except Exception as e:
                logging.warning(f"{case_id}: 分割转换失败（跳过）：{e}")
    else:
        logging.info(f"{case_id}: 未找到分割文件（允许无 GT）")

    # ---------- 5. 写 meta ----------
    write_meta(out_dir, ct_files, seg_src, choose_reason)

    elapsed = time.perf_counter() - start_t
    logging.info(f"{case_id}: DONE (⏱ {elapsed:.1f}s)")

def main():
    ap = argparse.ArgumentParser(description="Extract CT/SEG series → NIfTI + meta (standalone).")
    ap.add_argument("--cfg", default=None, help="可选：YAML 配置（需含 extract 段）")
    ap.add_argument("--raw_dicom_root", default=None, help="直接传参：原始 DICOM 根目录（优先于 cfg）")
    ap.add_argument("--raw_nifti_root", default=None, help="直接传参：NIfTI 输出根目录（优先于 cfg）")
    ap.add_argument("--patient_prefix", default=None, help="可选：仅处理这些前缀（用逗号分隔）")
    ap.add_argument("--seg_name_hint", default=None, help="可选：SEG 名称关键词（逗号分隔；默认 segmentation,result,seg）")
    ap.add_argument("--save_seg_as_dcm", action="store_true", help="SEG 转换失败时，保留 .dcm 备份")
    ap.add_argument("--overwrite", action="store_true", help="若目标已存在则先删除")
    ap.add_argument("--log-level", default="info",
                    choices=["debug", "info", "warning", "error", "critical"])
    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    cfg = load_cfg(args.cfg)
    # 解析参数优先于 cfg
    raw_dicom_root = Path(args.raw_dicom_root or (cfg["extract"]["raw_dicom_root"] if cfg else ""))
    raw_nifti_root = Path(args.raw_nifti_root or (cfg["extract"]["raw_nifti_root"] if cfg else ""))
    if not str(raw_dicom_root) or not str(raw_nifti_root):
        raise SystemExit("必须提供 --raw_dicom_root 与 --raw_nifti_root（或在 --cfg 的 extract 段内提供）")

    prefixes = []
    if args.patient_prefix:
        prefixes = [p.strip() for p in args.patient_prefix.split(",") if p.strip()]
    elif cfg and cfg["extract"].get("patient_prefix"):
        prefixes = cfg["extract"]["patient_prefix"]

    seg_name_hint = ["segmentation", "result", "seg"]
    if args.seg_name_hint:
        seg_name_hint = [s.strip() for s in args.seg_name_hint.split(",") if s.strip()]
    elif cfg and cfg["extract"].get("seg_name_hint"):
        seg_name_hint = cfg["extract"]["seg_name_hint"]

    save_seg_as_dcm = bool(args.save_seg_as_dcm or (cfg and cfg["extract"].get("save_seg_as_dcm", False)))

    if args.overwrite and raw_nifti_root.exists():
        shutil.rmtree(raw_nifti_root)
    raw_nifti_root.mkdir(parents=True, exist_ok=True)

    cases = [c for c in sorted(raw_dicom_root.iterdir()) if c.is_dir()]
    if prefixes:
        cases = [c for c in cases if any(c.name.startswith(p) for p in prefixes)]

    logging.info(f"Total cases: {len(cases)} | Output: {raw_nifti_root}")
    for c in cases:
        process_case(c, raw_nifti_root, seg_name_hint, save_seg_as_dcm)

if __name__ == "__main__":
    main()
