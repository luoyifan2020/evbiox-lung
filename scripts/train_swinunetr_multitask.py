# train_swinunetr_multitask.py

import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 加速设置
torch.backends.cudnn.benchmark = True

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from torch.cuda.amp import autocast, GradScaler

def main():
    # ─── 配置区 ─────────────────────────────────────────
    set_determinism(seed=42)
    ROOT         = Path(__file__).resolve().parent.parent
    DATALIST     = ROOT / "datalist.json"
    CLIN_CSV     = ROOT / "data" / "clinical" / "clinical.csv"
    ROI_SIZE     = (96, 96, 96)
    TASK_KEY     = "malignancy"
    TASK_OUT     = 1
    BATCH_SIZE   = 2
    NUM_WORKERS  = 4
    LR           = 1e-4
    MAX_EPOCHS   = 200
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ───────────────────────────────────────────────────────

    # 0. 载入临床表型
    clin_df = pd.read_csv(CLIN_CSV, index_col="patient_id")
    clin_map = clin_df[TASK_KEY].to_dict()

    # 1. 读 datalist.json 并附加临床标签
    with open(DATALIST, "r") as f:
        datadict   = json.load(f)
    train_files = datadict["training"]
    val_files   = datadict["validation"]
    def attach_clinical(files):
        for item in files:
            pid = Path(item["image"]).stem.replace("_img","")
            item[TASK_KEY] = float(clin_map.get(pid, 0.0))
    attach_clinical(train_files)
    attach_clinical(val_files)

    # 2. 定义 transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, mode='constant', constant_values=0),
        RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1,2)),
        ToTensord(keys=["image", "label", TASK_KEY]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, mode='constant', constant_values=0),
        CenterSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE),
        ToTensord(keys=["image", "label", TASK_KEY]),
    ])

    # 3. Dataset & DataLoader
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_num=24, cache_rate=1.0, num_workers=NUM_WORKERS
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_num=6, cache_rate=1.0, num_workers=NUM_WORKERS
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # 4. 模型
    class MultiTaskSwinUNETR(nn.Module):
        def __init__(self):
            super().__init__()
            self.segmenter = SwinUNETR(
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=False,
                spatial_dims=3,
            )

        def forward(self, x):
            seg_logits = self.segmenter(x)
            probs      = torch.softmax(seg_logits, 1)
            fg_prob    = probs[:, 1]
            reg_out    = fg_prob.mean(dim=(1,2,3), keepdim=True)
            reg_out    = reg_out.view(reg_out.shape[0], -1)
            return seg_logits, reg_out

    model = MultiTaskSwinUNETR().to(DEVICE)

    # 5. 损失 & 优化器 & 指标
    seg_loss_fn = DiceCELoss(include_background=True, to_onehot_y=True)
    reg_loss_fn = nn.L1Loss()
    metric      = DiceMetric(include_background=True, reduction="mean")
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler      = GradScaler()

    # 6. 训练 & 验证
    best_metric = -1.0
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs = batch["image"].to(DEVICE)
            lbls = batch["label"].to(DEVICE).long()
            cln  = batch[TASK_KEY].to(DEVICE).unsqueeze(1).float()

            optimizer.zero_grad()
            with autocast():
                seg_logits, reg_out = model(imgs)
                l_seg = seg_loss_fn(seg_logits, lbls)
                l_reg = reg_loss_fn(reg_out, cln)
                loss  = l_seg + 0.1 * l_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch} train loss: {epoch_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            metric.reset()
            for batch in tqdm(val_loader, desc="Validation"):
                imgs = batch["image"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                seg_logits, _ = model(imgs)
                val_seg = torch.softmax(seg_logits, 1)
                metric(y_pred=val_seg, y=lbls)
            m = metric.aggregate().item()

        print(f"Epoch {epoch} Validation Dice: {m:.4f}")
        if m > best_metric:
            best_metric = m
            torch.save(model.state_dict(), "best_swinunetr_multitask.pth")
            print("Saved new best model")

    print("Training completed. Best Dice:", best_metric)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
