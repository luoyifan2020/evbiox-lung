# ========================
# preprocess_image_gpu.py
# ========================
import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import torchio as tio
from pathlib import Path

# ———— 配置区 ————
ROOT     = Path(__file__).resolve().parent.parent
INTERIM  = ROOT / 'data' / 'interim' / 'lidc_preproc'
META_CSV = INTERIM / 'Meta' / 'meta_info.csv'
OUT_DIR  = ROOT / 'data' / 'processed' / 'step1_gpu'
PIXDIM   = (1.0, 1.0, 1.0)      # 重采样到 1×1×1 mm

# 创建输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义离线预处理 Transform：确定性操作
transform = tio.Compose([
    tio.ToCanonical(),  # 统一 RAS 方向
    tio.Resample(PIXDIM, image_interpolation='linear'),
    tio.Clamp(out_min=-1000.0, out_max=400.0),  # HU 截断
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

def main():
    # 读取所有 patient_id
    df = pd.read_csv(META_CSV)
    patients = sorted(df['patient_id'].unique())

    # 选用 GPU（或退回 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    for pid in patients:
        print(f"[{pid}] processing...")
        img_folder = INTERIM / 'Image' / pid
        seg_folder = INTERIM / 'Mask'  / pid
        # 如果根本没有该患者的 img 文件夹，直接跳过
        if not img_folder.exists():
            print(f"  !! no image folder for {pid}, skip")
            continue

        # 列出所有切片文件
        img_files = sorted(img_folder.glob(f"{pid}_*.npy"))
        seg_files = sorted(seg_folder.glob(f"{pid}_*.npy"))

        # 如果找不到任何切片，跳过
        if len(img_files) == 0:
            print(f"  !! no slices in {img_folder}, skip")
            continue
        # 可选：也可以同时检查 seg_files

        # 堆叠所有切片 → (D, H, W)
        img_arr = np.stack([np.load(str(p)) for p in img_files], axis=0)
        seg_arr = np.stack([np.load(str(p)) for p in seg_files], axis=0)

        # 构造 TorchIO Subject
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float().to(device)  # (1, D, H, W)
        seg_tensor = torch.from_numpy(seg_arr).unsqueeze(0).long().to(device)

        subj = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor, affine=np.eye(4)),
            label=tio.LabelMap(   tensor=seg_tensor, affine=np.eye(4))
        )

        # 执行离线 Transform
        out = transform(subj)
        img_out = out.image.data.cpu().numpy()[0]  # (D, H, W)
        seg_out = out.label.data.cpu().numpy()[0]  # (D, H, W)

        # 转换 dtype 并保存为 NIfTI
        img_cast = img_out.astype(np.float32)
        seg_cast = seg_out.astype(np.uint8)

        nib.save(
            nib.Nifti1Image(img_cast, affine=np.eye(4)),
            str(OUT_DIR / f"{pid}_img.nii.gz")
        )
        nib.save(
            nib.Nifti1Image(seg_cast, affine=np.eye(4)),
            str(OUT_DIR / f"{pid}_seg.nii.gz")
        )
        print(f"  → saved {pid}_img.nii.gz, {pid}_seg.nii.gz")

    print('Step1–3 GPU 离线处理完成.')

if __name__ == '__main__':
    main()
