#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_preprocess_nsclc_vessel.py (PA-only capable)
- 预处理(可复用已有 ct_preproc / lung_mask)
- 可选分割：lung_vessels / heart PA（及 PA∩Lung）
- 统计 & QC

用法同原脚本：
  python nsclc_swinunetr/scripts/02_preprocess_nsclc_vessel.py \
      --cfg nsclc_swinunetr/configs/preprocess_nsclc_vessel.yaml
"""

from __future__ import annotations
import argparse, json, logging, os, traceback, tempfile, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from nibabel import processing
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ---------------- utils ---------------- #
def _to_builtin(x: Any):
    import numpy as _np
    if isinstance(x, (_np.integer,)): return int(x)
    if isinstance(x, (_np.floating,)): return float(x)
    if isinstance(x, (_np.bool_,)): return bool(x)
    if isinstance(x, (list, tuple)): return [_to_builtin(i) for i in x]
    if isinstance(x, dict): return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, _np.ndarray): return x.tolist()
    return x

def dump_json(obj: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_builtin(obj), indent=2, ensure_ascii=False), encoding="utf-8")

_CT_CANDIDATES  = ["ct.nii.gz", "ct_nii.gz"]
_SEG_CANDIDATES = ["tumor_mask.nii.gz", "seg.nii.gz"]

def find_first(path: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = path / name
        if p.exists():
            return p
    return None

def bounding_box(mask: Optional[np.ndarray]) -> list[int] | None:
    if mask is None or not mask.any():
        return None
    idx = np.where(mask)
    return [int(arr.min()) for arr in idx] + [int(arr.max()) for arr in idx]

def resample(arr: np.ndarray, spacing: tuple[float,float,float],
             tgt_spacing: tuple[float,float,float], order: int) -> np.ndarray:
    aff = np.diag((*spacing, 1.0))
    img = nib.Nifti1Image(arr, aff)
    tgt_aff = np.diag((*tgt_spacing, 1.0))
    phys = np.array(arr.shape) * np.array(spacing)
    tgt_shape = np.maximum(np.round(phys / np.array(tgt_spacing)).astype(int), 1)
    tgt = (tgt_shape, tgt_aff)
    out = processing.resample_from_to(img, tgt, order=order)
    return out.get_fdata().astype(arr.dtype)

def pad_or_crop(arr: np.ndarray, new_shape: tuple[int,int,int]) -> np.ndarray:
    out = np.zeros(new_shape, dtype=arr.dtype)
    z = min(arr.shape[0], new_shape[0])
    y = min(arr.shape[1], new_shape[1])
    x = min(arr.shape[2], new_shape[2])
    out[:z, :y, :x] = arr[:z, :y, :x]
    return out

def bbox_crop(ct: np.ndarray, lung: np.ndarray, margin: int):
    if not lung.any():
        return ct, slice(0, ct.shape[0]), slice(0, ct.shape[1]), slice(0, ct.shape[2])
    z, y, x = np.where(lung)
    z0, z1 = max(int(z.min())-margin, 0), min(int(z.max())+margin+1, ct.shape[0])
    y0, y1 = max(int(y.min())-margin, 0), min(int(y.max())+margin+1, ct.shape[1])
    x0, x1 = max(int(x.min())-margin, 0), min(int(x.max())+margin+1, ct.shape[2])
    return ct[z0:z1, y0:y1, x0:x1], slice(z0, z1), slice(y0, y1), slice(x0, x1)

def save_qc_contour_png(
    ct_img: sitk.Image,
    vessel_img: sitk.Image,
    out_png: Path,
    lung_img: Optional[sitk.Image] = None,
    n_slices: int = 6,
    vessel_color="lime",
    lung_color="cyan",
    vessel_lw: float = 1.6,
    lung_lw: float = 1.0,
    qc_dilate_vox: int = 1,
):
    """在数个 z-slice 上叠加 mask 轮廓，方便快速肉眼审查。"""
    ct = sitk.GetArrayFromImage(ct_img)
    vs = sitk.GetArrayFromImage(vessel_img) > 0
    if qc_dilate_vox and qc_dilate_vox > 0 and vs.any():
        vs = ndi.binary_dilation(vs, iterations=int(qc_dilate_vox))
    lung = sitk.GetArrayFromImage(lung_img) > 0 if lung_img is not None else None

    z_has = np.where(vs.any(axis=(1, 2)))[0] if vs.any() else np.array([], dtype=int)
    if len(z_has) >= 2:
        z_idx = np.linspace(z_has[0], z_has[-1], n_slices, dtype=int)
    else:
        z_idx = np.linspace(0, ct.shape[0]-1, n_slices, dtype=int)

    lo, hi = np.percentile(ct, (0.5, 99.5))
    ct = np.clip((ct - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axs = plt.subplots(1, n_slices, figsize=(3.2 * n_slices, 3.4))
    if n_slices == 1:
        axs = [axs]
    for i, z in enumerate(z_idx):
        ax = axs[i]
        ax.imshow(ct[z], cmap="gray", origin="lower")
        if lung is not None and lung.any():
            ax.contour(lung[z].astype(float), levels=[0.5], colors=lung_color, linewidths=lung_lw)
        if vs.any():
            ax.contour(vs[z].astype(float), levels=[0.5], colors=vessel_color, linewidths=vessel_lw)
        ax.set_title(f"z={int(z)}")
        ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close(fig)

def ensure_img_like(msk: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """把分割图重采样到参考 CT 的网格上。"""
    return sitk.Resample(
        msk,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

def compute_stats(ct_img: sitk.Image, vs_img: sitk.Image, lung_img: Optional[sitk.Image] = None) -> Dict[str, Any]:
    spx, spy, spz = ct_img.GetSpacing()
    vs = sitk.GetArrayFromImage(vs_img) > 0
    vox = int(vs.sum())
    vol_mm3 = float(vox * spx * spy * spz)

    if vox > 0:
        z_has = np.where(vs.any(axis=(1, 2)))[0]
        z_range = [int(z_has[0]), int(z_has[-1])]
    else:
        z_range = [0, 0]

    lung_frac = None
    if lung_img is not None:
        lung = sitk.GetArrayFromImage(lung_img) > 0
        tot = int(lung.sum()) or 1
        lung_frac = float(vox) / float(tot)

    return dict(
        voxels=vox,
        volume_mm3=vol_mm3,
        volume_cm3=vol_mm3 / 1000.0,
        z_range=z_range,
        spacing_mm=[float(spx), float(spy), float(spz)],
        lung_fraction=lung_frac,
    )

# --------- 新增：根据“肺的 z 范围”清洗 PA 掩膜 --------- #
def clean_pa_by_lung_range(
    pa_img: sitk.Image,
    lung_img: Optional[sitk.Image],
    z_margin_slices: int = 5,
) -> sitk.Image:
    """
    只保留 z 轴位置落在 [lung_z_min - margin, lung_z_max + margin] 范围内的 PA 连通块，
    去掉远离肺的“奇怪块”（比如腹部被误分成 PA）。
    """
    if lung_img is None:
        return pa_img

    pa = sitk.GetArrayFromImage(pa_img) > 0
    if not pa.any():
        return pa_img

    lung = sitk.GetArrayFromImage(lung_img) > 0
    z_has = np.where(lung.any(axis=(1, 2)))[0]
    if len(z_has) == 0:
        return pa_img

    z_min_lu, z_max_lu = int(z_has[0]), int(z_has[-1])
    lo = max(z_min_lu - int(z_margin_slices), 0)
    hi = min(z_max_lu + int(z_margin_slices), pa.shape[0] - 1)

    lbl, num = ndi.label(pa)
    if num == 0:
        return pa_img

    keep = np.zeros_like(pa, dtype=bool)
    for lab in range(1, num + 1):
        idx = np.where(lbl == lab)
        if idx[0].size == 0:
            continue
        z_mean = idx[0].mean()
        if lo <= z_mean <= hi:
            keep[idx] = True

    pa_clean = sitk.GetImageFromArray(keep.astype(np.uint8))
    pa_clean.CopyInformation(pa_img)
    return pa_clean

# ---------------- segmentation backends ---------------- #
def run_lungmask(ct_path: Path, ct_img: nib.Nifti1Image) -> nib.Nifti1Image:
    from lungmask import mask as lungmask
    itk = sitk.ReadImage(str(ct_path))
    m_zyx = lungmask.apply(itk)  # Z,Y,X
    m_xyz = np.transpose(m_zyx, (2, 1, 0)).astype(np.uint8)
    return nib.Nifti1Image(m_xyz, ct_img.affine)

FAST_FORBIDDEN = {"lung_vessels"}

def run_totalseg(
    input_ct: Path,
    tmp_out: Path,
    task: str,
    try_fast: bool,
    cmd: str = "TotalSegmentator",
):
    args = [cmd, "-i", str(input_ct), "-o", str(tmp_out), "--task", task]
    if try_fast and (task not in FAST_FORBIDDEN):
        args.append("--fast")
    logging.info("[TotalSegmentator] " + " ".join(args))
    r = subprocess.run(args, capture_output=True, text=True)
    out_all = (r.stdout or "") + "\n" + (r.stderr or "")
    return r.returncode, out_all

def _find_any_nii(d: Path) -> Optional[Path]:
    cands = list(d.glob("*.nii*"))
    if not cands:
        return None

    def vox(p: Path):
        try:
            return np.prod(sitk.GetArrayFromImage(sitk.ReadImage(str(p))).shape)
        except Exception:
            return 0

    cands.sort(key=lambda p: vox(p), reverse=True)
    return cands[0]

def _find_pa_nii(d: Path) -> Optional[Path]:
    """优先找肺动脉；若确实不存在则退回到“最大体素数”的 nii。"""
    pats = ["pulmonary_artery", "artery_pulmonary", "pulmonary-artery", "pa"]
    for p in d.glob("*.nii*"):
        n = p.name.lower()
        if ("pulmon" in n and "artery" in n) or any(k in n for k in pats):
            return p
    return _find_any_nii(d)

def run_antspynet_pa(input_ct: Path, out_nifti: Path):
    try:
        import ants, antspynet
    except Exception as e:
        raise RuntimeError("ANTsPyNet 未安装，或 TF 版本不匹配。") from e
    ct = ants.image_read(str(input_ct))
    res = antspynet.lung_pulmonary_artery_segmentation(ct, verbose=True)
    seg = res.get("segmentation_image") if isinstance(res, dict) else res
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    ants.image_write(seg, tmp.name)
    sitk.WriteImage(sitk.ReadImage(tmp.name), str(out_nifti))

# ---------------- core pipeline ---------------- #
def process_case(case: Path, cfg: dict, vcfg: dict, num_threads: int) -> tuple[str, str]:
    pid = case.name
    tmp_ct_hu = None

    try:
        out_root = Path(cfg["preprocess_root"])
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        # 统一 QC 输出目录：R01-001/figs/*
        qc_dir = out_dir / vcfg.get("qc_subdir", "figs")

        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

        # 所有 case 下的原始 ct.nii
        ct_path = find_first(case, _CT_CANDIDATES)
        if ct_path is None:
            return pid, "missing_ct"

        # ---------- 复用预处理（若开关为真且文件存在） ----------
        reuse = bool(cfg.get("reuse_preprocessed", False))
        pre_ct = out_dir / "ct_preproc.nii.gz"
        pre_lu = out_dir / "lung_mask.nii.gz"

        if reuse and pre_ct.exists() and pre_lu.exists():
            ct_preproc_p = pre_ct
            lung_mask_p = pre_lu
            ct_img_ref = sitk.ReadImage(str(ct_preproc_p))
            lung_img_ref = sitk.ReadImage(str(lung_mask_p))

            input_for_ts = ct_preproc_p
            if vcfg.get("restore_hu_window", None) is not None:
                HMIN, HMAX = vcfg["restore_hu_window"]
                img = nib.load(str(ct_preproc_p))
                arr = img.get_fdata().astype(np.float32)
                arr_hu = arr * (HMAX - HMIN) + HMIN
                tmp_ct_hu = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                nib.save(nib.Nifti1Image(arr_hu.astype(np.int16), img.affine), tmp_ct_hu.name)
                input_for_ts = Path(tmp_ct_hu.name)

        else:
            # ---------- 正常预处理 ----------
            ct_img0 = nib.load(ct_path)
            ct_arr0 = ct_img0.get_fdata().astype(np.int16)
            spacing0 = ct_img0.header.get_zooms()[:3]

            # 肺分割（lungmask）
            lung_img0 = run_lungmask(ct_path, ct_img0)
            lung_img0 = processing.resample_from_to(lung_img0, ct_img0, order=0)
            lung_arr0 = lung_img0.get_fdata().astype(np.uint8)

            # 可选：把肿瘤 mask 合入肺域，避免 bbox 裁剪时漏掉
            seg_path = find_first(case, _SEG_CANDIDATES)
            tumor_arr0 = None
            if seg_path is not None:
                tumor_img0 = processing.resample_from_to(nib.load(seg_path), ct_img0, order=0)
                tumor_arr0 = (tumor_img0.get_fdata() > 0).astype(np.uint8)
                lung_arr0 = np.logical_or(lung_arr0, tumor_arr0).astype(np.uint8)

            # 重采样到目标 spacing
            tgt_sp = tuple(float(s) for s in cfg["target_spacing"])
            ct_rs    = resample(ct_arr0, spacing0, tgt_sp, order=3)
            lung_rs  = resample(lung_arr0, spacing0, tgt_sp, order=0)
            tumor_rs = resample(tumor_arr0, spacing0, tgt_sp, order=0) if tumor_arr0 is not None else None
            lung_rs  = pad_or_crop(lung_rs, ct_rs.shape)
            if tumor_rs is not None:
                tumor_rs = pad_or_crop(tumor_rs, ct_rs.shape)

            # bbox 裁剪（按肺的范围）
            if cfg.get("enable_crop", True):
                margin = cfg.get("crop_margin", 20)
                ct_rs, z, y, x = bbox_crop(ct_rs, lung_rs, margin)
                lung_rs = lung_rs[z, y, x]
                if tumor_rs is not None:
                    tumor_rs = tumor_rs[z, y, x]

            # HU window & 归一化到 [0,1]
            hmin, hmax = map(float, cfg["hu_window"])
            ct_rs = np.clip(ct_rs, hmin, hmax)
            ct_rs = (ct_rs - hmin) / (hmax - hmin)

            aff = np.diag((*tgt_sp, 1.0))
            nib.save(nib.Nifti1Image(ct_rs.astype(np.float32), aff), out_dir / "ct_preproc.nii.gz")
            nib.save(nib.Nifti1Image(lung_rs.astype(np.uint8),   aff), out_dir / "lung_mask.nii.gz")
            if tumor_rs is not None and tumor_rs.any():
                nib.save(nib.Nifti1Image(tumor_rs.astype(np.uint8), aff), out_dir / "_tumor_mask.nii.gz")

            ct_preproc_p = out_dir / "ct_preproc.nii.gz"
            lung_mask_p  = out_dir / "lung_mask.nii.gz"
            ct_img_ref   = sitk.ReadImage(str(ct_preproc_p))
            lung_img_ref = sitk.ReadImage(str(lung_mask_p))

            input_for_ts = ct_preproc_p
            if vcfg.get("restore_hu_window", None) is not None:
                HMIN, HMAX = vcfg["restore_hu_window"]
                img = nib.load(str(ct_preproc_p))
                arr = img.get_fdata().astype(np.float32)
                arr_hu = arr * (HMAX - HMIN) + HMIN
                tmp_ct_hu = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                nib.save(nib.Nifti1Image(arr_hu.astype(np.int16), img.affine), tmp_ct_hu.name)
                input_for_ts = Path(tmp_ct_hu.name)

        # ============== 1) 肺血管 lung_vessels（可选） ============== #
        if vcfg.get("enable_lung_vessels", True):
            out_main = out_dir / "vessel.nii.gz"
            out_png  = qc_dir / "qc_vessel.png"
            out_meta = out_dir / "vessel_meta.json"

            if (not out_main.exists()) or vcfg.get("overwrite", False):
                with tempfile.TemporaryDirectory() as td:
                    tmp_out = Path(td) / "ts_out"
                    tmp_out.mkdir(parents=True, exist_ok=True)

                    code, out_all = run_totalseg(
                        input_for_ts,
                        tmp_out,
                        task="lung_vessels",
                        try_fast=vcfg.get("fast", False),
                        cmd=vcfg.get("totalseg_cmd", "TotalSegmentator"),
                    )
                    if code != 0:
                        raise RuntimeError(f"TS lung_vessels 失败：\n{out_all}")

                    raw = _find_any_nii(tmp_out)
                    if raw is None:
                        raise RuntimeError("未找到肺血管 NIfTI")
                    vs_img = sitk.Cast(sitk.ReadImage(str(raw)) > 0.5, sitk.sitkUInt8)
                    sitk.WriteImage(ensure_img_like(vs_img, ct_img_ref), str(out_main))

            if out_main.exists():
                vs_img = sitk.ReadImage(str(out_main))
                dump_json(compute_stats(ct_img_ref, vs_img, lung_img_ref), out_meta)
                if not out_png.exists():
                    save_qc_contour_png(
                        ct_img_ref,
                        vs_img,
                        out_png,
                        lung_img_ref,
                        n_slices=vcfg.get("qc_slices", 6),
                        vessel_color=vcfg.get("qc_color", "lime"),
                        lung_color=vcfg.get("qc_lung_color", "cyan"),
                        vessel_lw=vcfg.get("qc_linewidth", 1.6),
                        qc_dilate_vox=vcfg.get("qc_dilate_vox", 1),
                    )

        # ============== 2) 肺动脉 / 心腔（TotalSegmentator + 清洗） ============== #
        if vcfg.get("enable_heart_pa", True):
            out_pa      = out_dir / "vessel_pa.nii.gz"
            out_pa_png  = qc_dir / "qc_vessel_pa.png"
            out_pa_meta = out_dir / "vessel_pa_meta.json"

            out_inlung      = out_dir / "vessel_pa_in_lung.nii.gz"
            out_inlung_png  = qc_dir / "qc_vessel_pa_in_lung.png"
            out_inlung_meta = out_dir / "vessel_pa_in_lung_meta.json"

            need = (not out_pa.exists()) or vcfg.get("overwrite", False)
            if need:
                ts_ok, ts_msg = False, ""
                with tempfile.TemporaryDirectory() as td:
                    tmp_out = Path(td) / "ts_out"
                    tmp_out.mkdir(parents=True, exist_ok=True)

                    code, out_all = run_totalseg(
                        input_for_ts,
                        tmp_out,
                        task="heartchambers_highres",   # 注意：此任务一般不建议 --fast
                        try_fast=vcfg.get("fast", False),
                        cmd=vcfg.get("totalseg_cmd", "TotalSegmentator"),
                    )

                    if code == 0:
                        # 2.1 备份所有心腔相关 nii -> _ts_pa
                        if vcfg.get("save_all_pa_parts", True):
                            dumpdir = out_dir / "_ts_pa"
                            dumpdir.mkdir(parents=True, exist_ok=True)
                            for p in tmp_out.glob("*.nii*"):
                                (dumpdir / p.name).write_bytes(p.read_bytes())

                        # 2.2 选取肺动脉 nii
                        pa_raw = _find_pa_nii(tmp_out)
                        if pa_raw and pa_raw.exists():
                            pa_img = sitk.Cast(sitk.ReadImage(str(pa_raw)) > 0.5, sitk.sitkUInt8)
                            # 先重采样到 ct_preproc 网格
                            pa_img = ensure_img_like(pa_img, ct_img_ref)
                            # 再按肺的 z 范围做一次连通域清洗
                            if vcfg.get("clean_pa_by_lung", True):
                                pa_img = clean_pa_by_lung_range(
                                    pa_img,
                                    lung_img_ref,
                                    z_margin_slices=vcfg.get("pa_z_margin_slices", 5),
                                )
                            sitk.WriteImage(pa_img, str(out_pa))
                            ts_ok = True
                    else:
                        ts_msg = out_all

                if not ts_ok:
                    if "license" in ts_msg.lower():
                        logging.info(f"[{pid}] heartchambers_highres 需要许可证；尝试 ANTsPyNet 回退")
                    try:
                        run_antspynet_pa(input_for_ts, out_pa)
                    except Exception as e:
                        raise RuntimeError(f"PA 分割失败（TS/回退均不可用）：{e}")

            # 统计 + QC：PA
            if out_pa.exists():
                pa_img = sitk.ReadImage(str(out_pa))
                dump_json(compute_stats(ct_img_ref, pa_img, lung_img_ref), out_pa_meta)
                if not out_pa_png.exists():
                    save_qc_contour_png(
                        ct_img_ref,
                        pa_img,
                        out_pa_png,
                        lung_img_ref,
                        n_slices=vcfg.get("qc_slices", 6),
                        vessel_color=vcfg.get("qc_color", "lime"),
                        lung_color=vcfg.get("qc_lung_color", "cyan"),
                        vessel_lw=vcfg.get("qc_linewidth", 1.6),
                        qc_dilate_vox=vcfg.get("qc_dilate_vox", 1),
                    )

                # 生成 PA ∩ Lung（默认使用略膨胀的肺）
                if lung_img_ref is not None:
                    inlung_dilate = int(vcfg.get("inlung_dilate_vox", 2))
                    if inlung_dilate > 0:
                        arr = sitk.GetArrayFromImage(lung_img_ref) > 0
                        arr = ndi.binary_dilation(arr, iterations=inlung_dilate)
                        lung_eff = sitk.GetImageFromArray(arr.astype(np.uint8))
                        lung_eff.CopyInformation(ct_img_ref)
                    else:
                        lung_eff = lung_img_ref

                    pa_bin = sitk.Cast(pa_img > 0, sitk.sitkUInt8)
                    lu_bin = sitk.Cast(lung_eff > 0, sitk.sitkUInt8)
                    in_lung = sitk.And(pa_bin, lu_bin)
                    sitk.WriteImage(ensure_img_like(in_lung, ct_img_ref), str(out_inlung))

                    dump_json(compute_stats(ct_img_ref, in_lung, lung_img_ref), out_inlung_meta)
                    if not out_inlung_png.exists():
                        save_qc_contour_png(
                            ct_img_ref,
                            in_lung,
                            out_inlung_png,
                            lung_img_ref,
                            n_slices=vcfg.get("qc_slices", 6),
                            vessel_color=vcfg.get("qc_color", "lime"),
                            lung_color=vcfg.get("qc_lung_color", "cyan"),
                            vessel_lw=vcfg.get("qc_linewidth", 1.6),
                            qc_dilate_vox=vcfg.get("qc_dilate_vox", 1),
                        )

        return pid, "OK"

    except Exception as e:
        logging.error(f"[{pid}] {e}")
        traceback.print_exc()
        return pid, f"err:{e}"

    finally:
        # 清理临时 HU 文件
        try:
            if tmp_ct_hu is not None:
                tmp_ct_hu.close()
                if os.path.exists(tmp_ct_hu.name):
                    os.unlink(tmp_ct_hu.name)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="nsclc_swinunetr/configs/preprocess_nsclc_vessel.yaml")
    ap.add_argument("--num_workers", type=int, default=max(os.cpu_count() // 2, 1))
    ap.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"])
    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    cfg_all = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    cfg  = cfg_all["preprocess"]
    vcfg = cfg_all["vessel"]

    raw_root = Path(cfg["raw_nifti_root"]).resolve()
    out_root = Path(cfg["preprocess_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = sorted([p for p in raw_root.iterdir() if find_first(p, _CT_CANDIDATES)])
    logging.info(f"Detected {len(cases)} cases | workers={args.num_workers}")

    results = []
    if args.num_workers <= 1:
        for c in tqdm(cases):
            results.append(process_case(c, cfg, vcfg, 1))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            fut = {ex.submit(process_case, c, cfg, vcfg, 1): c.name for c in cases}
            for f in tqdm(as_completed(fut), total=len(cases)):
                results.append(f.result())

    ok = sum(1 for _, s in results if s == "OK")
    errs = [(p, s) for p, s in results if s != "OK"]
    logging.info(f"Done. OK={ok}  FAIL={len(errs)}")
    if errs:
        (out_root / "errors.preprocess_vessel.txt").write_text(
            "\n".join(f"{p},{s}" for p, s in errs),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
