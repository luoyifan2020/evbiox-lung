#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_make_datalist_nsclc.py  — datalist 构建
- 遍历 ROI 目录，汇总 image/label/lung 等路径与 ROI 级统计（tumor_vox、ratio）
- 患者分层 K 折（按每个病人的平均 tumor_ratio 分箱）
- ★ 每个病人最多取 N 个 ROI（train/val 可分别设置；策略：largest 或 random）
- 导出：datalist_nsclc_fold{1..K}.json、datalist_nsclc_kfold.json、datalist_nsclc_test.json、patient_folds.csv、roi_manifest.csv
"""

import os, sys, json, yaml, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

def _find_first(d: Path, names: List[str]) -> Path | None:
    for n in names:
        p = d / n
        if p.exists():
            return p
    return None

def collect_rois(roi_root: Path,
                 include_fields: List[str],
                 min_ratio: float) -> Tuple[List[Dict], pd.DataFrame]:
    # 文件名约定
    req_map = {
        "image": ["image.nii.gz", "ct.nii.gz"],
        "label": ["label.nii.gz", "tumor.nii.gz", "mask.nii.gz"],
        "lung":  ["lung.nii.gz", "lung_mask.nii.gz"],
    }
    records = []
    rows = []
    for pid_dir in sorted([p for p in roi_root.iterdir() if p.is_dir()]):
        pid = pid_dir.name
        for d in sorted([p for p in pid_dir.iterdir() if p.is_dir()]):
            img_p = _find_first(d, req_map["image"])
            lbl_p = _find_first(d, req_map["label"])
            if img_p is None or lbl_p is None:
                continue
            # 可选字段
            extra = {}
            for f in include_fields:
                if f == "lung":
                    lp = _find_first(d, req_map["lung"])
                    extra["lung"] = str(lp) if lp else ""
            # 统计
            lbl_img = nib.load(str(lbl_p))
            lbl = (lbl_img.get_fdata() > 0.5).astype(np.uint8)
            vox = int(lbl.sum())
            ratio = float(vox) / float(lbl.size)
            if ratio < min_ratio:
                continue
            rec = {"pid": pid, "roi_id": d.name, "image": str(img_p), "label": str(lbl_p)}
            rec.update(extra)
            records.append(rec)
            rows.append({
                "pid": pid,
                "roi_id": d.name,
                "image": str(img_p),
                "label": str(lbl_p),
                "lung":  rec.get("lung", ""),
                "tumor_ratio_roi": ratio,
                "tumor_vox_roi": vox
            })
    manifest = pd.DataFrame(rows)
    return records, manifest

def patient_bins_from_manifest(manifest: pd.DataFrame, bins: List[float] | None):
    # 按病人聚合一个“平均前景比例”，再分箱
    g = manifest.groupby("pid")["tumor_ratio_roi"].mean()
    vals = g.values
    if not bins:
        # 3 档（小/中/大）
        q = np.quantile(vals, [0.33, 0.66]).tolist()
        bins = [-1e9] + q + [1e9]
    cats = pd.cut(g, bins=bins, labels=False, include_lowest=True)
    pid2bin = {pid: int(cats.loc[pid]) for pid in g.index}
    return pid2bin, bins

def kfold_by_patient_stratified(manifest: pd.DataFrame,
                                records: List[Dict],
                                k: int,
                                seed: int,
                                fold_bins: List[float] | None,
                                max_train_per_pid: int = 0,
                                max_val_per_pid: int = 0,
                                select_by: str = "largest"):
    rng = np.random.RandomState(seed)

    pid2bin, used_bins = patient_bins_from_manifest(manifest, fold_bins)
    pids = sorted(manifest["pid"].unique().tolist())

    # 把病人按 bin 分组后 round-robin 分到 k 折
    folds_pid = [set() for _ in range(k)]
    for b in sorted(set(pid2bin.values())):
        group = [pid for pid in pids if pid2bin[pid] == b]
        rng.shuffle(group)
        for i, pid in enumerate(group):
            folds_pid[i % k].add(pid)

    # 预聚合 ROI 记录（按病人）
    recs_by_pid: Dict[str, List[Dict]] = {}
    for r in records:
        recs_by_pid.setdefault(r["pid"], []).append(r)

    # 准备排序指标（largest 用）
    vox_map = {(r["pid"], r["roi_id"]): 0 for r in records}
    ratio_map = {(r["pid"], r["roi_id"]): 0.0 for r in records}
    for _, row in manifest.iterrows():
        vox_map[(row["pid"], row["roi_id"])] = int(row["tumor_vox_roi"])
        ratio_map[(row["pid"], row["roi_id"])] = float(row["tumor_ratio_roi"])

    def pick(pid_set: set, cap: int) -> List[Dict]:
        out = []
        for pid in sorted(pid_set):
            cand = recs_by_pid.get(pid, [])
            if cap and len(cand) > cap:
                if select_by.lower().startswith("rand"):
                    cand = rng.choice(cand, size=cap, replace=False).tolist()
                else:
                    cand = sorted(
                        cand,
                        key=lambda r: (vox_map.get((r["pid"], r["roi_id"]), 0),
                                       ratio_map.get((r["pid"], r["roi_id"]), 0.0)),
                        reverse=True
                    )[:cap]
            out.extend(cand)
        return out

    folds = []
    for i in range(k):
        val_pids = folds_pid[i]
        train_pids = set().union(*[folds_pid[j] for j in range(k) if j != i])
        tr = pick(train_pids, max_train_per_pid)
        va = pick(val_pids,   max_val_per_pid)
        folds.append({"train": tr, "val": va})

    # 患者-折次表
    pf_rows = []
    for i in range(k):
        for pid in sorted(folds_pid[i]):
            pf_rows.append({"pid": pid, "fold": i + 1, "bin": pid2bin[pid]})
    patient_folds = pd.DataFrame(pf_rows)
    return folds, patient_folds, used_bins

def export_datalist_json(out_path: Path, name: str, training: List[Dict], validation: List[Dict]):
    out = {"name": name, "training": training, "validation": validation}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

def export_test_json(out_path: Path, name: str, items: List[Dict]):
    # 测试清单统一放在 "samples" 下；评估脚本支持 "testing" 或 "samples"
    out = {"name": name, "samples": items}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="datalist_nsclc.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    dcfg = cfg.get("datalist", cfg)

    roi_root  = Path(dcfg["roi_root"])
    out_dir   = Path(dcfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    name      = str(dcfg.get("name", "NSCLC_ROI"))
    include_fields = dcfg.get("include_fields", ["lung"])
    min_ratio = float(dcfg.get("min_tumor_ratio", 0.0))

    k     = int(dcfg.get("k", 5))
    seed  = int(dcfg.get("seed", 42))
    fold_bins = dcfg.get("fold_bins", None)  # 例如 [0, 1e-5, 5e-3, 1.0]
    export_all_folds = bool(dcfg.get("export_all_folds", True))
    export_test_as_datalist = bool(dcfg.get("export_test_as_datalist", True))
    holdout_ratio = float(dcfg.get("holdout_ratio", 0.15))

    max_train_per_pid = int(dcfg.get("max_rois_per_pid_train", 0))
    max_val_per_pid   = int(dcfg.get("max_rois_per_pid_val", 0))
    select_by         = str(dcfg.get("select_by", "largest"))

    # 1) 收集 ROI
    records, manifest = collect_rois(roi_root, include_fields, min_ratio)
    manifest_path = out_dir / "roi_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"[manifest] {manifest_path} | rows={len(manifest)}")

    # 2) 划分 train/val/test（病人级）
    pids = sorted(manifest["pid"].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n_hold = int(round(len(pids) * holdout_ratio))
    test_pids = set(pids[:n_hold])
    trainval_pids = set(pids[n_hold:])
    print(f"[split] trainval_pids={len(trainval_pids)} | test_pids={len(test_pids)}")

    folds, patient_folds, used_bins = kfold_by_patient_stratified(
        manifest=manifest[~manifest["pid"].isin(test_pids)].copy(),
        records=[r for r in records if r["pid"] in trainval_pids],
        k=k, seed=seed, fold_bins=fold_bins,
        max_train_per_pid=max_train_per_pid,
        max_val_per_pid=max_val_per_pid,
        select_by=select_by
    )
    pf_path = out_dir / "patient_folds.csv"
    patient_folds.to_csv(pf_path, index=False, encoding="utf-8-sig")
    print(f"[patient_folds] {pf_path}")

    # 3) 导出 k 折 datalist
    kfold_all = {"name": f"{name}_kfold", "folds": []}
    for i, fold in enumerate(folds, 1):
        fold_path = out_dir / f"datalist_nsclc_fold{i}.json"
        export_datalist_json(fold_path, f"{name}_fold{i}", fold["train"], fold["val"])
        kfold_all["folds"].append({
            "name": f"fold{i}",
            "training": fold["train"],
            "validation": fold["val"]
        })
        print(f"[fold{i}] train={len(fold['train'])}, val={len(fold['val'])} -> {fold_path}")
    (out_dir / "datalist_nsclc_kfold.json").write_text(json.dumps(kfold_all, indent=2), encoding="utf-8")

    # 4) 导出 test datalist
    test_records = [r for r in records if r["pid"] in test_pids]
    test_path = out_dir / "datalist_nsclc_test.json"
    export_test_json(test_path, f"{name}_test", test_records)
    print(f"[test] {len(test_records)} -> {test_path}")

if __name__ == "__main__":
    main()
