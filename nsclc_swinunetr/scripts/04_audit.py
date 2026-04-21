import argparse
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm
from functools import partial

def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return numpy array."""
    return nib.load(str(path)).get_fdata()


def audit_one(roi_dir: Path, cube_size: int | None = None) -> dict:
    """
    Audit a single ROI folder.

    期望目录结构（来自 03_roi_builder）:
        <roi_root>/<pid>/<roi_id>/
            image.nii.gz  # cropped CT cube
            lung.nii.gz   # lung mask in cube
            label.nii.gz  # tumor mask in cube
    """
    roi_dir = Path(roi_dir)
    pid = roi_dir.parent.name
    roi_id = roi_dir.name

    ct_p = roi_dir / "image.nii.gz"
    lung_p = roi_dir / "lung.nii.gz"
    tum_p = roi_dir / "label.nii.gz"

    res: dict[str, object] = {
        "pid": pid,
        "roi_id": roi_id,
        "roi_path": str(roi_dir),
        "status": "",
        "ct_exists": int(ct_p.exists()),
        "lung_exists": int(lung_p.exists()),
        "tumor_exists": int(tum_p.exists()),
        "cube_size": cube_size if cube_size is not None else "",
        "shape": "",
        "cube_ok": "",
        "vox_ct": "",
        "vox_lung": "",
        "vox_tumor": "",
        "tumor_ratio_roi": "",
        "tumor_ratio_lung": "",
        "tumor_outside_lung_ratio": "",
    }

    # 必要文件检查
    if not ct_p.exists() or not lung_p.exists():
        res["status"] = "MISSING_CORE"  # CT 或 lung mask 缺失
        return res

    # 加载数据
    ct = load_nifti(ct_p)
    lung = load_nifti(lung_p) > 0
    if tum_p.exists():
        tumor = load_nifti(tum_p) > 0
    else:
        tumor = np.zeros_like(lung, dtype=bool)

    res["shape"] = str(ct.shape)

    # 形状检查（是否立方体 & 尺寸）
    if cube_size is not None:
        cube_ok = (
            ct.shape[0] == cube_size
            and ct.shape[1] == cube_size
            and ct.shape[2] == cube_size
        )
        res["cube_ok"] = int(cube_ok)
        if not cube_ok:
            res["status"] = "SHAPE_MISMATCH"
    else:
        res["cube_ok"] = ""

    vox_ct = int(np.prod(ct.shape))
    vox_lung = int(lung.sum())
    vox_tumor = int(tumor.sum())
    res["vox_ct"] = vox_ct
    res["vox_lung"] = vox_lung
    res["vox_tumor"] = vox_tumor

    # 若完全无瘤体素
    if vox_tumor == 0:
        if not res["status"]:
            res["status"] = "NO_TUMOR"
        return res

    # ROI 内瘤体占比
    tumor_ratio_roi = float(vox_tumor / vox_ct)
    res["tumor_ratio_roi"] = tumor_ratio_roi

    # 肺内瘤体 & 肺外瘤体
    tumor_in_lung = np.logical_and(tumor, lung).sum()
    tumor_outside = vox_tumor - int(tumor_in_lung)
    tumor_ratio_lung = float(tumor_in_lung / vox_lung) if vox_lung > 0 else 0.0
    outside_ratio = float(tumor_outside / vox_tumor)

    res["tumor_ratio_lung"] = tumor_ratio_lung
    res["tumor_outside_lung_ratio"] = outside_ratio

    # 状态判定
    status_flags: list[str] = []

    # 瘤体太小（可以后续根据分布再调）
    if tumor_ratio_roi < 5e-4:      # <0.05% of cube
        status_flags.append("VERY_SMALL")
    elif tumor_ratio_roi < 5e-3:   # <0.5% of cube
        status_flags.append("SMALL")

    # 大部分瘤体都在肺外
    if outside_ratio > 0.2:
        status_flags.append("OUTSIDE_LUNG")

    if not status_flags:
        status_flags.append("OK")

    # 如果前面已有 SHAPE_MISMATCH 等，拼接 flag
    if res["status"]:
        status_flags.insert(0, res["status"])

    res["status"] = "+".join(status_flags)
    return res


def load_cfg(cfg_path: str | None) -> dict:
    if not cfg_path:
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_summary(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.cfg)
    audit_cfg = cfg.get("audit", {})

    roi_root = Path(
        args.roi_root or audit_cfg.get("roi_root", "data/roi/nsclc")
    )
    out_csv = Path(
        audit_cfg.get("out_csv", "audit_roi.csv")
    )
    cube_size = args.cube_size or audit_cfg.get("cube_size", None)
    num_workers = args.num_workers or audit_cfg.get("num_workers", 4)

    if cube_size is not None:
        cube_size = int(cube_size)

    # 发现 ROI 目录：按 image.nii.gz 反推父目录
    roi_niis = list(roi_root.rglob("image.nii.gz"))
    roi_dirs = sorted({p.parent for p in roi_niis})

    if not roi_dirs:
        print(f"[WARN] 在 {roi_root} 下没有找到任何 ROI (image.nii.gz).")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pid",
                    "roi_id",
                    "roi_path",
                    "status",
                    "ct_exists",
                    "lung_exists",
                    "tumor_exists",
                    "cube_size",
                    "shape",
                    "cube_ok",
                    "vox_ct",
                    "vox_lung",
                    "vox_tumor",
                    "tumor_ratio_roi",
                    "tumor_ratio_lung",
                    "tumor_outside_lung_ratio",
                ]
            )
        print(f"[INFO] 已写入空的审计表头: {out_csv}")
        return

    print(f"[INFO] 将审计 {len(roi_dirs)} 个 ROI 目录，root={roi_root}")
    rows: list[dict] = []

    # 预先把 cube_size “固化” 进去，生成一个可被 pickle 的顶层函数包装
    worker = partial(audit_one, cube_size=cube_size)

    if num_workers is None or num_workers <= 1:
        # 单进程版本，调试时可以把 num_workers 设为 1
        for res in tqdm(map(worker, roi_dirs), total=len(roi_dirs)):
            rows.append(res)
    else:
        # 多进程版本（Windows 也 OK，因为没有 lambda 了）
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(ex.map(worker, roi_dirs), total=len(roi_dirs)):
                rows.append(res)


    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pid",
        "roi_id",
        "roi_path",
        "status",
        "ct_exists",
        "lung_exists",
        "tumor_exists",
        "cube_size",
        "shape",
        "cube_ok",
        "vox_ct",
        "vox_lung",
        "vox_tumor",
        "tumor_ratio_roi",
        "tumor_ratio_lung",
        "tumor_outside_lung_ratio",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] 审计完成，共 {len(rows)} 条记录，写入: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser("Audit ROI quality (NSCLC, EVBioX-Lung).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_sum = sub.add_parser("summary", help="对所有 ROI 生成审计 CSV")
    s_sum.add_argument("--cfg", help="audit_roi.yaml 配置文件路径")
    s_sum.add_argument("--roi_root", help="覆盖 cfg 中的 audit.roi_root")
    s_sum.add_argument("--cube_size", type=int, help="ROI 立方体尺寸（如 96 或 128）")
    s_sum.add_argument("--num_workers", type=int, help="并行进程数")
    s_sum.set_defaults(func=run_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
