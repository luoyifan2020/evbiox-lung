#!/usr/bin/env python
# scripts/prepare_dataset.py

import os
import sys
from pathlib import Path
from configparser import ConfigParser
import pandas as pd
import numpy as np

# 兼容旧代码对 np.int 和 np.bool 的调用
setattr(np, 'int', int)
setattr(np, 'bool', bool)

import warnings
warnings.filterwarnings(action='ignore')

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# LIDC 预处理脚本所在目录
LIDC_PREPROC_DIR = ROOT / 'externals' / 'lidc_preproc'

# 读配置文件
parser = ConfigParser()
parser.read(str(LIDC_PREPROC_DIR / 'lung.conf'))

# 从配置获取 DICOM 原始数据路径，设置环境变量供 pylidc 使用
DICOM_DIR = parser.get('prepare_dataset', 'LIDC_DICOM_PATH')
os.environ['LIDC_IDRI_PATH'] = DICOM_DIR

# 把 externals/lidc_preproc 加到 sys.path，这样可以直接 import utils
sys.path.insert(0, str(LIDC_PREPROC_DIR))
from utils import is_dir_path, segment_lung

import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm
from statistics import median_high
from PIL import Image

# 从配置获取各个目录
MASK_DIR         = is_dir_path(parser.get('prepare_dataset', 'MASK_PATH'))
IMAGE_DIR        = is_dir_path(parser.get('prepare_dataset', 'IMAGE_PATH'))
CLEAN_DIR_IMAGE  = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK   = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
META_DIR         = is_dir_path(parser.get('prepare_dataset', 'META_PATH'))

# 超参数
mask_threshold   = parser.getint('prepare_dataset', 'Mask_Threshold')
confidence_level = parser.getfloat('pylidc', 'confidence_level')
padding_size     = parser.getint('pylidc', 'padding_size')

class MakeDataSet:
    def __init__(self, patient_list, image_dir, mask_dir,
                 clean_image_dir, clean_mask_dir, meta_dir,
                 mask_threshold, padding_size, confidence_level=0.5):
        self.patient_list   = patient_list
        self.img_root       = Path(image_dir)
        self.mask_root      = Path(mask_dir)
        self.clean_img_root = Path(clean_image_dir)
        self.clean_mask_root= Path(clean_mask_dir)
        self.meta_path      = Path(meta_dir)
        self.mask_threshold = mask_threshold
        self.c_level        = confidence_level
        self.padding        = [(padding_size, padding_size),
                               (padding_size, padding_size),
                               (0, 0)]
        self.meta = pd.DataFrame(
            columns=[
                'patient_id', 'nodule_no', 'slice_no',
                'original_image', 'mask_image',
                'malignancy', 'is_cancer', 'is_clean'
            ]
        )

    def calculate_malignancy(self, nodule):
        scores = [ann.malignancy for ann in nodule]
        mscore = median_high(scores)
        if mscore > 3:
            return mscore, True
        elif mscore < 3:
            return mscore, False
        else:
            return mscore, 'Ambiguous'

    def save_meta(self, meta_list):
        row = pd.Series(meta_list, index=self.meta.columns)
        self.meta.loc[len(self.meta)] = row

    def prepare_dataset(self):
        prefix = [str(i).zfill(3) for i in range(1000)]
        # 确保所有目录存在
        for path in [
            self.img_root, self.mask_root,
            self.clean_img_root, self.clean_mask_root,
            self.meta_path
        ]:
            path.mkdir(parents=True, exist_ok=True)

        for pid in tqdm(self.patient_list, desc='Patients'):
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            if scan is None:
                print(f"[WARNING] 无法找到扫描数据 {pid}，跳过")
                continue

            annotations = scan.cluster_annotations()
            volume      = scan.to_volume()
            print(f"Patient {pid}: volume shape {volume.shape}, nodules {len(annotations)}")

            img_dir  = self.img_root / pid
            mask_dir = self.mask_root / pid
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            if annotations:
                for n_idx, nodule in enumerate(annotations):
                    mask_vol, bbox, _ = consensus(nodule, self.c_level, self.padding)
                    cropped           = volume[bbox]
                    malignancy, is_cancer = self.calculate_malignancy(nodule)

                    for s_idx in range(mask_vol.shape[2]):
                        mask_slice = mask_vol[:, :, s_idx]
                        if np.sum(mask_slice) <= self.mask_threshold:
                            continue
                        img_slice = segment_lung(cropped[:, :, s_idx])
                        img_slice[img_slice == -0] = 0

                        img_name  = f"{pid}_NI{prefix[n_idx]}_slice{prefix[s_idx]}"
                        mask_name = f"{pid}_MA{prefix[n_idx]}_slice{prefix[s_idx]}"
                        meta_entry = [
                            pid, n_idx, s_idx,
                            img_name, mask_name,
                            malignancy, is_cancer, False
                        ]

                        self.save_meta(meta_entry)
                        np.save(img_dir  / img_name,  img_slice)
                        np.save(mask_dir / mask_name, mask_slice)
            else:
                # 无标注，存负样本
                print(f"[CLEAN] {pid} 无结节标注，将保存为负样本")
                clean_img_dir  = self.clean_img_root / pid
                clean_mask_dir = self.clean_mask_root / pid
                clean_img_dir.mkdir(parents=True, exist_ok=True)
                clean_mask_dir.mkdir(parents=True, exist_ok=True)

                max_slices = min(volume.shape[2], 50)
                for s_idx in range(max_slices):
                    img_slice  = segment_lung(volume[:, :, s_idx])
                    img_slice[img_slice == -0] = 0
                    mask_slice = np.zeros_like(img_slice)

                    cn_name = f"{pid}_CN001_slice{prefix[s_idx]}"
                    cm_name = f"{pid}_CM001_slice{prefix[s_idx]}"
                    meta_entry = [
                        pid, None, s_idx,
                        cn_name, cm_name,
                        0, False, True
                    ]

                    self.save_meta(meta_entry)
                    np.save(clean_img_dir  / cn_name,  img_slice)
                    np.save(clean_mask_dir / cm_name, mask_slice)

        out_csv = self.meta_path / 'meta_info.csv'
        self.meta.to_csv(out_csv, index=False)
        print(f"Metadata saved to {out_csv}")

if __name__ == '__main__':
    # 枚举所有 LIDC-IDRI 开头的患者
    patients = sorted([d for d in os.listdir(DICOM_DIR) if d.startswith('LIDC-IDRI')])
    dataset = MakeDataSet(
        patient_list     = patients,
        image_dir        = IMAGE_DIR,
        mask_dir         = MASK_DIR,
        clean_image_dir  = CLEAN_DIR_IMAGE,
        clean_mask_dir   = CLEAN_DIR_MASK,
        meta_dir         = META_DIR,
        mask_threshold   = mask_threshold,
        padding_size     = padding_size,
        confidence_level = confidence_level
    )
    dataset.prepare_dataset()
