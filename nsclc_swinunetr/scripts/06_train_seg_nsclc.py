# -*- coding: utf-8 -*-
"""
06_train_seg_nsclc.py (改良版)
- 验证频率与保存频率解耦：val_interval 控制验证；save_interval 仅控制额外快照
- 每个 epoch 都可打印 val_best_dice
- 训练/验证使用 non_blocking 拷贝；cudnn.benchmark & matmul precision
- 验证集缓存从 cfg.data.val_cache_rate 读取（默认 0.0）
"""

import os, sys, json, math, time, copy, random, argparse, traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ---- 环境静音 & CUDA 分配器 ----
import warnings
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# 不设置无效 TORCH_LOGS；若外部设了，清理之
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_LOGS_ARTIFACTS", None)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
try:
    from setuptools import SetuptoolsDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=SetuptoolsDeprecationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module=r"pkg_resources")
warnings.filterwarnings("ignore", module=r"setuptools")
warnings.filterwarnings("ignore", module=r"torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import gc
gc.collect()
torch.cuda.empty_cache()

import monai
from monai.data import CacheDataset, DataLoader, Dataset
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, CenterSpatialCropd,
    RandFlipd, RandRotate90d, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, RandShiftIntensityd, SpatialPadd,
    RandCropByPosNegLabeld,
    )
from monai.utils import set_determinism
from inspect import signature

import os, torch, torch.backends.cudnn as cudnn
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")  # 避免一次性加载所有 CUDA DLL
cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# ----- 通用工具 -----
def set_all_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)

def logi(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mean_dice_binary(preds: torch.Tensor, labels: torch.Tensor, ignore_empty: bool = True, eps: float = 1e-6) -> float:
    preds = preds.detach(); labels = labels.detach()
    N = preds.shape[0]; dices = []
    for i in range(N):
        p = preds[i, 0].reshape(-1); y = labels[i, 0].reshape(-1)
        if ignore_empty and torch.sum(y) == 0:
            continue
        inter = torch.sum(p * y)
        d = (2.0 * inter + eps) / (torch.sum(p) + torch.sum(y) + eps)
        dices.append(d.item())
    return 0.0 if len(dices)==0 else float(np.mean(dices))

# ----- 自定义变换 -----
class HUWindowd(monai.transforms.MapTransform):
    def __init__(self, keys, hu_min=-1000, hu_max=400):
        super().__init__(keys); self.hu_min=float(hu_min); self.hu_max=float(hu_max)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            img = d[k].astype(np.float32)
            img = np.clip(img, self.hu_min, self.hu_max)
            img = (img - self.hu_min) / max(1e-6, (self.hu_max - self.hu_min))
            d[k] = img
        return d

class SafeConcatLungd(monai.transforms.MapTransform):
    def __init__(self, image_key="image", lung_key="lung", out_key="image"):
        super().__init__([image_key, lung_key])
        self.image_key=image_key; self.lung_key=lung_key; self.out_key=out_key
    def _center_pad_or_crop(self, arr, target_shape):
        a = arr
        if a.ndim == 3: a = a[None, ...]
        C,H,W,D = a.shape; th,tw,td = target_shape[-3:]
        ph=max(0,th-H); pw=max(0,tw-W); pd=max(0,td-D)
        if ph or pw or pd:
            a = np.pad(a, ((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(pd//2,pd-pd//2)), mode="constant")
        C,H,W,D = a.shape; sh=max(0,H-th); sw=max(0,W-tw); sd=max(0,D-td)
        if sh or sw or sd:
            a = a[:, sh//2:sh//2+th, sw//2:sw//2+tw, sd//2:sd//2+td]
        if arr.ndim == 3: a = a[0]
        return a
    def __call__(self, data):
        d = dict(data); img = d.get(self.image_key, None); lung = d.get(self.lung_key, None)
        if img is None: return d
        if isinstance(img, torch.Tensor): img = img.numpy()
        if lung is None:
            if img.ndim == 3: img = img[None, ...]; d[self.out_key]=img; return d
        if isinstance(lung, torch.Tensor): lung = lung.numpy()
        if img.ndim == 3: img = img[None, ...]
        if lung.ndim == 3: lung = lung[None, ...]
        lung = (lung > 0).astype(np.float32)
        lung = self._center_pad_or_crop(lung, img.shape)
        d[self.out_key] = np.concatenate([img, lung], axis=0)
        return d

# ----- 损失 -----
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.gamma=gamma; self.eps=eps
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        tp = (probs * target).sum(dim=(2,3,4))
        fp = (probs * (1-target)).sum(dim=(2,3,4))
        fn = ((1-probs) * target).sum(dim=(2,3,4))
        tversky = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps)
        return torch.pow((1.0 - tversky), self.gamma).mean()

class ComboLoss(nn.Module):
    def __init__(self, dice_w=0.5, ftl_w=0.5, bce_w=0.0, ftl_args=None, bce_pos=1.0):
        super().__init__()
        self.dice_w = dice_w; self.ftl_w = ftl_w; self.bce_w = bce_w
        self.dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ftl = FocalTverskyLoss(**(ftl_args or {"alpha":0.7,"beta":0.3,"gamma":0.75}))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(bce_pos)))
    def forward(self, logits, target):
        def _single(x):
            loss = 0.0
            if self.dice_w>0: loss += self.dice_w*self.dice(x, target)
            if self.ftl_w>0:  loss += self.ftl_w*self.ftl(x, target)
            if self.bce_w>0:  loss += self.bce_w*self.bce(x, target)
            return loss
        if isinstance(logits, (list, tuple)):
            total = 0.0
            for i, x in enumerate(logits):
                total += (1.0 if i == 0 else 0.4) * _single(x)
            return total
        
        return _single(logits)


# ----- 模型构建 -----
def build_model(cfg: dict, in_channels: int, out_channels: int):
    mcfg = cfg.get("model", {})
    img_size = tuple(mcfg.get("img_size", cfg.get("roi_size", [128,128,128])))
    feature_size = int(mcfg.get("feature_size", 48))
    depths = mcfg.get("depths", [2,2,2,2])
    num_heads = mcfg.get("num_heads", [3,6,12,24])
    dropout_path_rate = float(mcfg.get("dropout_path_rate", 0.0))
    use_checkpoint = bool(mcfg.get("use_checkpoint", False))
    spatial_dims = int(mcfg.get("spatial_dims", 3))
    norm_name = mcfg.get("norm_name", "instance")
    downsample = mcfg.get("downsample", "merging")
    qkv_bias = bool(mcfg.get("qkv_bias", True))
    mlp_drop_rate = float(mcfg.get("mlp_drop_rate", 0.0))
    attn_drop_rate = float(mcfg.get("attn_drop_rate", 0.0))

    sig_params = set(signature(SwinUNETR).parameters.keys())
    kwargs = {
        "img_size": img_size, "in_channels": in_channels, "out_channels": out_channels,
        "feature_size": feature_size, "depths": depths, "num_heads": num_heads,
        "dropout_path_rate": dropout_path_rate, "use_checkpoint": use_checkpoint,
        "spatial_dims": spatial_dims, "norm_name": norm_name, "downsample": downsample,
        "qkv_bias": qkv_bias, "mlp_drop_rate": mlp_drop_rate, "attn_drop_rate": attn_drop_rate,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
    model = SwinUNETR(**kwargs)
    return model

def _state_dict_filter(state: Dict[str,torch.Tensor], model: nn.Module) -> Dict[str,torch.Tensor]:
    msd = model.state_dict(); out = {}
    for k, w in state.items():
        if k not in msd: continue
        tgt = msd[k]
        if w.shape == tgt.shape:
            out[k] = w; continue
        # 通道数自适应（如 1 -> 2）
        if w.ndim==5 and tgt.ndim==5 and w.shape[0]==tgt.shape[0] and w.shape[2:]==tgt.shape[2:]:
            in_src, in_tgt = w.shape[1], tgt.shape[1]
            if in_src==1 and in_tgt>1:
                out[k] = w.repeat(1, in_tgt, 1,1,1) / in_tgt
            elif in_src>1 and in_tgt==1:
                out[k] = w[:, :1, ...].contiguous()
    return out

def load_ckpt_flex(model: nn.Module, ckpt_path: str, map_location="cpu") -> int:
    if not ckpt_path or not Path(ckpt_path).exists(): return 0
    logi(f"[*] load pretrain (flex): {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=map_location)
    if "state_dict" in sd: sd = sd["state_dict"]
    ok_sd = _state_dict_filter(sd, model)
    missing, unexpected = model.load_state_dict(ok_sd, strict=False)
    logi(f"    loaded: {len(ok_sd)} | missing={len(missing)} | unexpected={len(unexpected)}")
    return len(ok_sd)

def split_enc_dec_params(model: nn.Module):
    enc, dec = [], []
    for n, p in model.named_parameters():
        if "swinViT" in n: enc.append(p)
        else: dec.append(p)
    return enc, dec

# ----- 优化器 & 调度 -----
def build_optimizer_scheduler(model, cfg):
    optcfg = cfg.get("optim", {})
    lr = float(optcfg.get("lr", 1e-4))
    wd = float(optcfg.get("weight_decay", 8e-5))
    enc_ratio = float(optcfg.get("enc_lr_ratio", 0.5))

    enc, dec = split_enc_dec_params(model)
    param_groups = [
        {"params": dec, "lr": lr, "weight_decay": wd},
        {"params": enc, "lr": lr * enc_ratio, "weight_decay": wd},
    ]
    opt_name = str(optcfg.get("name", "adamw")).lower()
    optimr = optim.AdamW(param_groups, lr=lr, weight_decay=wd) if opt_name!="adam" \
             else optim.Adam(param_groups, lr=lr, weight_decay=wd)

    schecfg = cfg.get("scheduler", {"type": "cosine_warmup"})
    stype = str(schecfg.get("type", "cosine_warmup")).lower()
    epochs = int(cfg.get("epochs", 160))

    if stype == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimr, mode="max",
            factor=float(schecfg.get("factor", 0.5)),
            patience=int(schecfg.get("patience", 5)),
            min_lr=float(schecfg.get("min_lr", 1e-7)),
        )
    elif stype == "poly":
        power = float(schecfg.get("power", 0.9))
        total_steps = max(1, epochs)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimr, lr_lambda=lambda e: (1 - e/float(total_steps))**power
        )
    else:
        warmup = int(schecfg.get("warmup_epochs", 5))
        min_lr = float(schecfg.get("min_lr", 1e-6))
        base_lr = lr
        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            t = (epoch - warmup) / max(1, (epochs - warmup))
            return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * t)))
        scheduler = optim.lr_scheduler.LambdaLR(optimr, lr_lambda)
    return optimr, scheduler

# ----- 数据读取 & 变换 -----
def load_datalist(path: str) -> Tuple[List[dict], List[dict]]:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    return j.get("training", []), j.get("validation", [])

def build_transforms(cfg, in_channels: int):
    dcfg = cfg["data"]
    patch = tuple(dcfg.get("patch_size", [128, 128, 128]))
    hu_min, hu_max = dcfg.get("hu_window", [-1000, 400])
    use_lung = bool(dcfg.get("use_lung_channel", False))
    lung_key = dcfg.get("lung_key", "lung")

    flip_p   = float(dcfg.get("flip_prob", 0.3))
    rot90_p  = float(dcfg.get("rot90_prob", 0.3))
    contr_p  = float(dcfg.get("contr_prob", 0.15))
    shift_p  = float(dcfg.get("shift_prob", 0.15))
    shift_r  = float(dcfg.get("shift_range", 0.07))
    noise_p  = float(dcfg.get("noise_prob", 0.0))
    smooth_p = float(dcfg.get("smooth_prob", 0.0))

    base_keys = ["image", "label"] + ([lung_key] if use_lung else [])

    train_tf = [
        LoadImaged(keys=base_keys),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        HUWindowd(keys=["image"], hu_min=hu_min, hu_max=hu_max),
        SpatialPadd(keys=base_keys, spatial_size=patch),
        CenterSpatialCropd(keys=base_keys, roi_size=patch),
        RandCropByPosNegLabeld(
            keys=base_keys, label_key="label", spatial_size=patch,
            pos=dcfg.get("pos", 2), neg=dcfg.get("neg", 0),
            num_samples=dcfg.get("num_samples", 2), image_key="image", image_threshold=0.0,
        ),
        RandFlipd(keys=base_keys, prob=flip_p, spatial_axis=0),
        RandFlipd(keys=base_keys, prob=flip_p, spatial_axis=1),
        RandFlipd(keys=base_keys, prob=flip_p, spatial_axis=2),
        RandRotate90d(keys=base_keys, prob=rot90_p, max_k=3),
        RandAdjustContrastd(keys=["image"], prob=contr_p, gamma=(0.7, 1.3)),
        RandShiftIntensityd(keys=["image"], prob=shift_p, offsets=(-shift_r, shift_r)),
    ]
    if noise_p > 0:
        train_tf.append(RandGaussianNoised(keys=["image"], prob=noise_p, mean=0.0, std=0.05))
    if smooth_p > 0:
        train_tf.append(RandGaussianSmoothd(keys=["image"], prob=smooth_p, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)))
    train_tf.append(EnsureTyped(keys=base_keys))
    if use_lung:
        train_tf.append(SafeConcatLungd(image_key="image", lung_key=lung_key, out_key="image"))

    val_tf = [
        LoadImaged(keys=base_keys),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        HUWindowd(keys=["image"], hu_min=hu_min, hu_max=hu_max),
        SpatialPadd(keys=base_keys, spatial_size=patch),
        CenterSpatialCropd(keys=base_keys, roi_size=patch),
        EnsureTyped(keys=base_keys),
    ]
    if use_lung:
        val_tf.append(SafeConcatLungd(image_key="image", lung_key=lung_key, out_key="image"))
    return Compose(train_tf), Compose(val_tf)

# ----- 验证 -----
@torch.no_grad()
def validate(model: nn.Module, loader, device, cfg) -> Dict[str,Any]:
    model.eval()
    vcfg = cfg["val"]
    roi = tuple(vcfg.get("roi_size", cfg["data"].get("patch_size",[128,128,128])))
    sw_batch = int(vcfg.get("sw_batch_size", 1))
    overlap = float(vcfg.get("sw_overlap", 0.3))
    thr_list = list(vcfg.get("thresholds",[0.3,0.4,0.5,0.6,0.7]))
    dices = {thr: [] for thr in thr_list}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = sliding_window_inference(
            images, roi_size=roi, sw_batch_size=sw_batch,
            predictor=model, overlap=overlap, mode="gaussian"
        )
        probs = torch.sigmoid(logits)
        for thr in thr_list:
            preds = (probs > thr).float()
            d = mean_dice_binary(preds, labels, ignore_empty=True)
            dices[thr].append(d)
    avg = {thr: float(np.mean(v)) for thr, v in dices.items()}
    best_thr = max(avg, key=avg.get)
    return {"dices": avg, "best_thr": best_thr, "best_dice": avg[best_thr]}

# ----- 训练主流程 -----
def train(cfg: Dict[str,Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    seed = int(cfg.get("seed",42)); set_all_seeds(seed)

    # datalist
    dfile = cfg["data"]["datalist"]
    train_list, val_list = load_datalist(dfile)
    logi(f"data: train={len(train_list)} | val={len(val_list)}")

    # transforms
    use_lung = bool(cfg["data"].get("use_lung_channel", False))
    in_ch = 2 if use_lung else 1
    train_tf, val_tf = build_transforms(cfg, in_ch)

    # datasets / loaders
    cache_rate = float(cfg["data"].get("cache_rate", 0.0))
    val_cache_rate = float(cfg["data"].get("val_cache_rate", 0.0))  # 新增：验证缓存
    num_workers = int(cfg.get("loader_workers", 4))
    pin_memory  = bool(cfg.get("pin_memory", False))
    persistent  = bool(cfg.get("persistent_workers", False))
    prefetch    = int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    train_ds = CacheDataset(train_list, transform=train_tf, cache_rate=cache_rate, num_workers=0) \
               if cache_rate > 0 else Dataset(train_list, transform=train_tf)

    if val_cache_rate > 0:
        val_ds = CacheDataset(val_list, transform=val_tf, cache_rate=val_cache_rate, num_workers=0)
    else:
        val_ds = Dataset(val_list, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.get("batch_size", 1)), shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
        prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
        prefetch_factor=prefetch
    )

    logi(f"[env] device={device.type} | amp={bool(cfg.get('amp', True))}")
    logi(f"[loader] workers={num_workers} | pin_memory={pin_memory} | persistent={persistent} | prefetch_factor={prefetch}")
    logi(f"[batch] train_bs={cfg.get('batch_size', 1)} | val_bs=1 | patch={tuple(cfg['data'].get('patch_size',[128,128,128]))}")

    # 预取一批（val）以预热
    try:
        _ = next(iter(val_loader))
        logi("[prefetch] val_loader 1 batch ok; begin training soon...")
    except Exception as e:
        logi(f"[prefetch] skipped: {e}")

    # model / pretrain
    model = build_model(cfg, in_channels=in_ch, out_channels=1).to(device)
    if cfg.get("pretrain", None) and not cfg.get("resume", None):
        load_ckpt_flex(model, cfg["pretrain"], map_location="cpu")

    # optim / sched
    optimr, scheduler = build_optimizer_scheduler(model, cfg)

    # freeze encoder (optional)
    freeze_epochs = int(cfg.get("optim",{}).get("freeze_encoder_epochs", 0))
    def set_encoder_requires_grad(req: bool):
        for n,p in model.named_parameters():
            if "swinViT" in n: p.requires_grad = req
    if freeze_epochs > 0:
        logi(f"[*] freeze encoder for {freeze_epochs} epochs")
        set_encoder_requires_grad(False)

    # resume
    start_epoch = 0; best_dice = -1.0; best_thr = 0.5
    if cfg.get("resume", None) and Path(cfg["resume"]).exists():
        ckpt = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimr.load_state_dict(ckpt["optim"])
        if "sched" in ckpt and hasattr(scheduler,"load_state_dict"):
            try: scheduler.load_state_dict(ckpt["sched"])
            except Exception: pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", -1)); best_thr  = float(ckpt.get("best_thr", 0.5))
        logi(f"[*] resume from {cfg['resume']} @epoch {start_epoch} (best_dice={best_dice:.4f}, best_thr={best_thr:.2f})")

    # loss & amp
    lcfg = cfg["loss"]
    loss_fn = ComboLoss(
        dice_w=float(lcfg.get("weights",[0.6,0.6,0.05])[0]),
        ftl_w=float(lcfg.get("weights",[0.6,0.6,0.05])[1]),
        bce_w=float(lcfg.get("weights",[0.6,0.6,0.05])[2]),
        ftl_args=lcfg.get("ftl", {"alpha":0.7,"beta":0.3,"gamma":0.75}),
        bce_pos=float(lcfg.get("bce", {"pos_weight":3.0}).get("pos_weight",3.0))
    ).to(device)
    use_amp = bool(cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # out / metrics
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text("epoch,train_loss,val_best_dice,val_best_thr,lr\n", encoding="utf-8")

    # 训练超参（新版：val/save 解耦）
    epochs = int(cfg.get("epochs", 160))
    val_every  = int(cfg.get("val_interval", 1))           # <= 新增：验证间隔（默认每个 epoch）
    save_every = int(cfg.get("save_interval", 0))          # <= 仅用于额外快照，0 表示不额外存
    grad_clip = float(cfg.get("grad_clip", 0.0))
    accumulate = int(cfg.get("grad_accum_steps", 1))
    log_every = int(cfg.get("log_every", 50))

    logi("[start] training loop begin")
    global_step = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logi(f"=== Epoch {epoch+1}/{epochs} start ===")
        model.train()
        if freeze_epochs>0 and epoch==freeze_epochs:
            logi("[*] unfreeze encoder"); set_encoder_requires_grad(True)

        running = 0.0; optimr.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                if logits.shape[1] != 1: logits = logits[:,0:1,...]
                loss = loss_fn(logits, labels) / accumulate
            scaler.scale(loss).backward()
            if (it+1) % accumulate == 0:
                if grad_clip>0:
                    scaler.unscale_(optimr)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimr); scaler.update(); optimr.zero_grad(set_to_none=True)
            running += loss.item()*accumulate; global_step += 1
            if (it+1) % max(1,log_every) == 0:
                lr = optimr.param_groups[0]["lr"]
                logi(f"  iter {it+1}/{len(train_loader)} | loss={running/(it+1):.4f} | lr={lr:.2e}")

        # scheduler step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            pass
        else:
            scheduler.step()
        train_loss = running / max(1, len(train_loader))

        # 验证（按 val_every）
        do_val = ((epoch + 1) % max(1, val_every) == 0) or (epoch == epochs - 1)
        if do_val:
            vres = validate(model, val_loader, device, {
                "val": cfg.get("val", {}),
                "data": {"patch_size": cfg["data"].get("patch_size", [128,128,128])}
            })
            val_best_dice = vres["best_dice"]; val_best_thr = vres["best_thr"]
            # 更新 best
            if val_best_dice > best_dice:
                best_dice = val_best_dice; best_thr = val_best_thr
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optim": optimr.state_dict(),
                    "sched": scheduler.state_dict() if hasattr(scheduler,"state_dict") else {},
                    "best_dice": best_dice, "best_thr": best_thr, "cfg": cfg
                }, out_dir / "best.pth")
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_best_dice)
        else:
            val_best_dice = float("nan"); val_best_thr = float("nan")

        # 保存 last
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optim": optimr.state_dict(),
            "sched": scheduler.state_dict() if hasattr(scheduler,"state_dict") else {},
            "best_dice": best_dice, "best_thr": best_thr, "cfg": cfg
        }, out_dir / "last.pth")

        # 额外快照（仅按 save_every）
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optim": optimr.state_dict(),
                "sched": scheduler.state_dict() if hasattr(scheduler,"state_dict") else {},
                "best_dice": best_dice, "best_thr": best_thr, "cfg": cfg
            }, out_dir / f"epoch_{epoch+1:03d}.pth")

        lr = optimr.param_groups[0]["lr"]
        metrics_csv.open("a", encoding="utf-8").write(
            f"{epoch},{train_loss:.6f},{val_best_dice:.6f},{val_best_thr},{lr:.8f}\n"
        )
        logi(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_best_dice={val_best_dice:.4f} | best(dice={best_dice:.4f},thr={best_thr:.2f}) | lr={lr:.2e} | time={(time.time()-t0):.1f}s")

    logi(f"Done. Best dice={best_dice:.4f} @thr={best_thr:.2f} | out_dir={out_dir}")

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    cfg_text = Path(args.cfg).read_text(encoding="utf-8")
    if args.cfg.endswith(".json"):
        cfg = json.loads(cfg_text)
    else:
        import yaml
        cfg = yaml.safe_load(cfg_text)
    try:
        train(cfg)
    except Exception as e:
        logi(f"[ERROR] {e}")
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()
