# make_datalist.py
import json
import random
import pandas as pd
from pathlib import Path

# —— 配置区 —— 
ROOT        = Path(__file__).resolve().parent.parent
STEP1_DIR   = ROOT / 'data' / 'processed' / 'step1_gpu'
CLIN_CSV    = ROOT / 'data' / 'clinical'  / 'clinical.csv'
OUT_PATH    = ROOT / 'datalist.json'
TRAIN_RATIO = 0.8
# ——————————

# 调试：确认输入目录
print(f"STEP1_DIR: {STEP1_DIR}")
print(f"Exists:     {STEP1_DIR.exists()}")

# 1. 列出所有 _img.nii.gz 文件
img_paths = sorted(STEP1_DIR.glob('*_img.nii.gz'))
print(f"Found {len(img_paths)} files matching '*_img.nii.gz'")
for p in img_paths[:5]:
    print("  sample image:", p.name)

# 2. 构建 image/label 对列表
data_pairs = []
for img in img_paths:
    # 用文件名替换的方式提取 pid，避免 stem 多去掉一次 .nii
    # 例: "LIDC-IDRI-0001_img.nii.gz" --> pid="LIDC-IDRI-0001"
    pid = img.name.replace('_img.nii.gz', '')
    seg = STEP1_DIR / f"{pid}_seg.nii.gz"
    if not seg.exists():
        print(f"  !! no seg for {img.name}, expected {seg.name}, skipping")
        continue
    entry = {
        'image': str(img),
        'label': str(seg),
    }
    # 如果有临床表型，可在此读取并加入；此处先注释掉以确认配对
    # if pid in clin_map:
    #     entry.update(clin_map[pid])
    data_pairs.append(entry)

print(f"After pairing: {len(data_pairs)} samples")

# 3. 随机打乱 & 划分 train/validation
random.seed(42)
random.shuffle(data_pairs)
split = int(len(data_pairs) * TRAIN_RATIO)
train = data_pairs[:split]
val   = data_pairs[split:]

# 4. 写出 datalist.json
datalist = {'training': train, 'validation': val}
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(datalist, f, indent=2, ensure_ascii=False)

print(f"datalist.json 生成完成：训练={len(train)}，验证={len(val)}")
