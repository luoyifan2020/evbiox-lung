# -*- coding: utf-8 -*-
"""
17_fusion_report_allin1_strict.py  (OOF-only + optional TEST, pretty plot + robust clean)
- 面向“16/19 风险级 GMoE 的输出目录（oof_dir）”，默认只汇总/可视化 OOF（val）部分。
- 通过 oof_split 显式控制参与 OOF 统计/绘图的 split（默认仅 'val'）。
- 其它功能：IBS、Bootstrap、门控权重汇总、coverage、drop-one 轻量重训、可选 TEST 汇总。
"""
from __future__ import annotations
import argparse, json, math, re, warnings, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"]   = 110
plt.rcParams["savefig.dpi"]  = 220
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

# ---------------- utils ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _rotate_xtick_labels(ax=None, angle: int = 35, bottom_pad: float = 0.20):
    ax = ax or plt.gca()
    for lab in ax.get_xticklabels():
        lab.set_rotation(angle)
        lab.set_horizontalalignment("right")
        lab.set_rotation_mode("anchor")
    fig = ax.figure
    try:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, float(bottom_pad)))
    except Exception:
        pass

def _clean_pid(s: str) -> str:
    s = str(s).strip().replace("_","-").upper()
    s = re.sub(r"[\u2010-\u2015\u2212\uFF0D]", "-", s)
    s = re.sub(r"[^A-Z0-9\-]+","",s)
    s = re.sub(r"-{2,}","-",s)
    return s

def _pick_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() in ("pid","case_id","id","subject","subject_id"): return c
    for c in df.columns:
        if "pid" in str(c).lower(): return c
    return df.columns[0]

def _palette(n:int, cmap:str="viridis"):
    cmap_=plt.get_cmap(cmap, max(2,n))
    return [cmap_(i) for i in range(max(2,n))][:n]

def _safe_savefig(out_png: Path):
    try:
        plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight")
    except Exception:
        try:
            fig=plt.gcf(); w,h=fig.get_size_inches()
            fig.set_size_inches(min(max(w,6),16), min(max(h,4),20))
            plt.savefig(out_png)
        except Exception as e2:
            print(f"[WARNING] savefig fallback failed: {e2}")
    finally:
        plt.close()

def _clean_for_metrics(df: pd.DataFrame, cols=("time","risk","event"), name="oof") -> pd.DataFrame:
    """将指定列转为数值，替换 inf，并丢弃含 NaN/inf 的行，避免 lifelines 报错。"""
    before = len(df)
    dfx = df.copy()
    for c in cols:
        if c not in dfx.columns:
            continue
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dfx = dfx.dropna(subset=[c for c in cols if c in dfx.columns])
    dropped = before - len(dfx)
    if dropped > 0:
        print(f"[clean] {name}: dropped {dropped} rows due to NaN/inf in {cols}.")
    return dfx

def set_seed(seed:int=2025):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- subjects/folds ----------------
def load_subjects(mm_dir: str) -> pd.DataFrame:
    subs = pd.read_csv(Path(mm_dir)/"subjects.csv")
    if "pid" not in subs.columns:
        cid=_pick_id_col(subs); subs=subs.rename(columns={cid:"pid"})
    subs["pid"]=subs["pid"].astype(str).map(_clean_pid)
    ren={}
    for c in subs.columns:
        lc=str(c).lower()
        if lc in ("os_time","time_days"): ren[c]="time"
        if lc in ("os_event","status","event_observed"): ren[c]="event"
    if ren: subs=subs.rename(columns=ren)
    assert {"pid","time","event"}.issubset(subs.columns), "subjects.csv 需要含 pid/time/event"
    return subs[["pid","time","event"]].copy()

def load_folds(mm_dir: str) -> pd.DataFrame:
    folds = pd.read_csv(Path(mm_dir)/"folds.csv")
    if "pid" not in folds.columns:
        cid=_pick_id_col(folds); folds=folds.rename(columns={cid:"pid"})
    folds["pid"]=folds["pid"].astype(str).map(_clean_pid)
    if "split" not in folds.columns: folds["split"]=np.where(folds["fold"]==-1,"test","train")
    folds["fold"]=pd.to_numeric(folds["fold"], errors="coerce").astype(int)
    return folds[["pid","fold","split"]].copy()

# ---------------- read risks for coverage ----------------
def read_risk_csv(path: str, key: str) -> pd.DataFrame:
    df=pd.read_csv(path)
    if "pid" not in df.columns:
        df=df.rename(columns={_pick_id_col(df):"pid"})
    df["pid"]=df["pid"].astype(str).map(_clean_pid)
    ren={}
    for c in df.columns:
        lc=str(c).lower()
        if lc in ("os_time","time_days"): ren[c]="time"
        if lc in ("os_event","status","event_observed"): ren[c]="event"
        if lc=="risk": ren[c]=f"risk_{key}"
    if ren: df=df.rename(columns=ren)
    return df

# ---------------- KM / IBS ----------------
def km_plot(oof: pd.DataFrame, out_png: Path, title="KM (OOF)") -> float:
    from lifelines import KaplanMeierFitter
    oof=_clean_for_metrics(oof, cols=("time","risk","event"), name="oof@km")
    thr=oof["risk"].median()
    hi=oof[oof["risk"]>=thr]; lo=oof[oof["risk"]<thr]
    p=float(logrank_test(hi["time"], lo["time"], hi["event"].astype(bool), lo["event"].astype(bool)).p_value)
    km=KaplanMeierFitter(); plt.figure(figsize=(8.0,5.0))
    cmap=_palette(2,"viridis")
    for (g,d),color in zip([("High risk",hi),("Low risk",lo)], cmap):
        km.fit(d["time"].values, d["event"].astype(bool).values, label=f"{g} (n={len(d)})")
        km.plot_survival_function(ci_show=False, color=color, lw=2.0)
    plt.title(f"{title}  p={p:.2g}"); plt.xlabel("Time"); plt.ylabel("Survival probability")
    _safe_savefig(out_png)
    return p

def fit_breslow(train_df: pd.DataFrame):
    df=train_df.sort_values("time").reset_index(drop=True)
    eta=df["risk"].values - df["risk"].mean()
    hr=np.exp(np.clip(eta,-20,20)); t=df["time"].values; e=df["event"].astype(int).values
    uniq=np.unique(t[e==1]); H0=[]; cum=0.0
    for tau in uniq:
        d=((t==tau)&(e==1)).sum(); at=hr[t>=tau].sum()
        if at>0: cum+=d/at
        H0.append(cum)
    return uniq, np.array(H0,float), float(df["risk"].mean())

def survival_on_grid(val_df, uniq_t, H0, mu, grid):
    if len(uniq_t)==0: H0g=np.zeros_like(grid)
    else: H0g=np.interp(grid, uniq_t, H0, left=0.0, right=H0[-1])
    eta=val_df["risk"].values - mu; hr=np.exp(np.clip(eta,-20,20))
    S=np.exp(-np.outer(hr,H0g))
    return np.clip(S,1e-6,1-1e-6)

def ibs_fold(train_df, val_df, m=120):
    if len(train_df)==0 or len(val_df)==0: return float("nan"),None,None
    uniq_t,H0,mu=fit_breslow(train_df)
    grid=np.linspace(0, max(val_df["time"].max(), uniq_t[-1] if len(uniq_t) else 1.0), m)
    S=survival_on_grid(val_df, uniq_t, H0, mu, grid)
    e=val_df["event"].astype(int).values; t=val_df["time"].values
    I=(grid.reshape(-1,1)>=t.reshape(1,-1)).astype(float)
    brier = ((I - S.T)**2).mean(axis=1)
    ibs = float(np.trapz(brier, grid)/grid[-1])
    return ibs, grid, brier

# ---------------- coef/gate/coverage ----------------
def plot_coef_summary(oof: pd.DataFrame, use_keys: List[str], out_dir: Path):
    rec=[]
    for k in use_keys:
        arr=oof.get(f"w_{k}", pd.Series([])).replace([np.inf,-np.inf],np.nan).dropna().values
        if arr.size==0: continue
        rec.append({"feature":f"w_{k}","mean":float(arr.mean()), "std":float(arr.std())})
    if not rec:
        print("[coef] 没有可汇总的 w_*，跳过"); return
    df=pd.DataFrame(rec).sort_values("mean",ascending=False)
    df.to_csv(out_dir/"coef_summary.csv", index=False)

    plt.figure(figsize=(8, 3.8 + 0.28*len(df))); y=np.arange(len(df))
    colors=_palette(len(df), "viridis")
    plt.barh(y, df["mean"], xerr=df["std"], align="center", color=colors, alpha=0.95)
    plt.yticks(y, df["feature"]); plt.gca().invert_yaxis()
    plt.xlabel("Gate weight (mean ± SD)")
    _safe_savefig(out_dir/"coef_bar.png")

    plt.figure(figsize=(8.6,5.6))
    data=[oof.get(f"w_{k}", pd.Series([])).replace([np.inf,-np.inf],np.nan).dropna().values for k in use_keys]
    plt.boxplot(data, labels=[f"w_{k}" for k in use_keys], showfliers=True)
    plt.ylabel("Gate weight"); plt.title("Modality gate weights (OOF)")
    _rotate_xtick_labels(angle=35, bottom_pad=0.22)
    _safe_savefig(out_dir/"gate_box.png")

    means=[np.nanmean(x) if len(x) else np.nan for x in data]
    plt.figure(figsize=(8,4))
    plt.bar([f"w_{k}" for k in use_keys], means, color=_palette(len(means),"viridis"))
    plt.ylabel("Mean gate weight"); plt.title("Gate mean (OOF)")
    _rotate_xtick_labels(angle=35, bottom_pad=0.20)
    _safe_savefig(out_dir/"gate_bar.png")

# ---------------- bootstrap ----------------
def bootstrap_cindex(oof: pd.DataFrame, out_dir: Path, n=2000, seed=2025):
    set_seed(seed)
    arr=oof[["time","risk","event"]].to_numpy()
    N=len(arr); vals=[]
    for _ in range(int(n)):
        idx=np.random.randint(0,N,size=N)
        t,r,e=arr[idx,0], arr[idx,1], arr[idx,2]
        vals.append(float(concordance_index(t, -r, e)))
    vals=np.array(vals,float); ci=np.quantile(vals, [0.025,0.5,0.975])
    pd.DataFrame({"cindex":vals}).to_csv(out_dir/"bootstrap_cindex.csv", index=False)

    plt.figure(figsize=(7.2,4.5))
    plt.hist(vals, bins=40, color="#4C78A8", alpha=0.95)
    plt.axvline(ci[1], color="#E64B35", lw=2, label=f"median={ci[1]:.3f}")
    plt.axvline(ci[0], color="#EECA3B", ls="--", label=f"2.5%={ci[0]:.3f}")
    plt.axvline(ci[2], color="#EECA3B", ls="--", label=f"97.5%={ci[2]:.3f}")
    plt.legend(); plt.title("Bootstrap C-index")
    _safe_savefig(out_dir/"bootstrap_hist.png")
    return ci

# ---------------- drop-one（轻量） ----------------
class RiskDS(Dataset):
    """显式用 float32，避免 DataLoader -> 模型 forward 时出现 double/float 冲突。"""
    def __init__(self, X, M, T, E):
        self.X = np.asarray(X, dtype=np.float32)
        self.M = np.asarray(M, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32)
        self.E = np.asarray(E, dtype=np.float32)

    def __len__(self):
        return len(self.T)

    def __getitem__(self, i):
        import torch
        return (
            torch.from_numpy(self.X[i]).float(),
            torch.from_numpy(self.M[i]).float(),
            torch.tensor(self.T[i], dtype=torch.float32),
            torch.tensor(self.E[i], dtype=torch.float32),
        )

class GateNet(nn.Module):
    def __init__(self, m: int, hid: int = 64, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(m + m, hid), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(hid, hid),   nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(hid, m)
        )

    def forward(self, x, m):
        # 统一 dtype，避免 float/double 冲突
        x = x.float()
        m = m.float()
        g = self.mlp(torch.cat([x, m], dim=1))
        # 对缺失模态做强屏蔽（softmax 前加巨大负数）
        g = g + (m - 1.0) * 1e9
        return torch.softmax(g, dim=1)

def light_train_ablation(base: pd.DataFrame, use_keys: List[str], cfg: dict, out_dir: Path):
    set_seed(int(cfg.get("seed", 2025)))
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    folds    = sorted(base["fold"].unique().tolist())
    lr       = float(cfg.get("lr", 1e-3))
    bs       = int(cfg.get("batch", 128))
    epoch    = int(cfg.get("epochs", 120))
    patience = int(cfg.get("patience", 20))
    hid      = int(cfg.get("hid", 64))
    drop     = float(cfg.get("drop", 0.10))

    def _fit(keys: List[str]) -> float:
        cols = [f"risk_{k}" for k in keys]

        # numpy 侧先统一 float32，并把 NaN/Inf 变 0
        X = base[cols].to_numpy(dtype=np.float32, copy=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        M = (~np.isnan(base[cols].to_numpy())).astype(np.float32)  # 按“是否提供该模态的risk”构造 mask
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        T = pd.to_numeric(base["time"], errors="coerce").to_numpy().astype(np.float32)
        E = pd.to_numeric(base["event"], errors="coerce").to_numpy().astype(np.float32)

        cv = []
        for k in folds:
            TR = (base["fold"] != k).values
            VA = (base["fold"] == k).values

            Xtr, Mtr, Ttr, Etr = X[TR], M[TR], T[TR], E[TR]
            Xva, Mva, Tva, Eva = X[VA], M[VA], T[VA], E[VA]

            net = GateNet(len(keys), hid, drop).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)

            best, bad = 1e9, 0
            for ep in range(epoch):
                # -------- train --------
                net.train(); n = 0; tr = 0.0
                loader = DataLoader(RiskDS(Xtr, Mtr, Ttr, Etr), batch_size=bs, shuffle=True, drop_last=False)
                for xb, mb, tb, eb in loader:
                    xb, mb, tb, eb = xb.to(device), mb.to(device), tb.to(device), eb.to(device)
                    opt.zero_grad()
                    g = net(xb, mb)                 # [B, m]
                    r = (g * xb).sum(1)             # [B]
                    order = torch.argsort(tb, descending=False)
                    rr = r[order]; ee = eb[order]
                    # Cox-NLL（无 ties 简化）
                    loss = -(rr - torch.logcumsumexp(rr, 0)) * ee
                    loss = loss.sum() / (ee.sum() + 1e-8)
                    loss.backward(); opt.step()
                    tr += float(loss.item()); n += 1

                # -------- val --------
                net.eval(); va = 0.0; n = 0
                with torch.no_grad():
                    xv = torch.tensor(Xva, dtype=torch.float32, device=device)
                    mv = torch.tensor(Mva, dtype=torch.float32, device=device)
                    tv = torch.tensor(Tva, dtype=torch.float32, device=device)
                    ev = torch.tensor(Eva, dtype=torch.float32, device=device)
                    g  = net(xv, mv)
                    r  = (g * xv).sum(1)
                    order = torch.argsort(tv, descending=False)
                    rr = r[order]; ee = ev[order]
                    loss = -(rr - torch.logcumsumexp(rr, 0)) * ee
                    loss = loss.sum() / (ee.sum() + 1e-8)
                    va += float(loss.item()); n += 1

                va /= max(1, n)
                if va + 1e-6 < best:
                    best = va; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience: break

            if best < 1e9:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                xv = torch.tensor(Xva, dtype=torch.float32, device=device)
                mv = torch.tensor(Mva, dtype=torch.float32, device=device)
                rv = (net(xv, mv) * xv).sum(1).float().cpu().numpy()
                # 再保险：将任何 NaN/Inf 置 0
                rv = np.nan_to_num(rv, nan=0.0, posinf=0.0, neginf=0.0)

            v = base.loc[VA, ["time", "event"]].copy()
            v["time"]  = pd.to_numeric(v["time"], errors="coerce")
            v["event"] = pd.to_numeric(v["event"], errors="coerce")
            v["risk"]  = rv
            cv.append(v)

        oof = pd.concat(cv, axis=0, ignore_index=True) if cv else pd.DataFrame()
        oof = _clean_for_metrics(oof, ("time", "risk", "event"), name="ablation-oof")
        return float(concordance_index(oof["time"], -oof["risk"], oof["event"])) if len(oof) else float("nan")

    base_keys = list(use_keys) if len(use_keys) else list(cfg.get("use", []))
    if len(base_keys) < 2:
        print("[ablation] 仅一个模态，跳过。"); return

    c_base = _fit(base_keys)
    rows = []
    for d in base_keys:
        keep = [k for k in base_keys if k != d]
        c = _fit(keep)
        rows.append({"dropped": d, "cindex_oof": c, "delta_vs_base": c_base - c})

    df = pd.DataFrame(rows).sort_values("cindex_oof", ascending=False)
    df.to_csv(out_dir / "ablation_drop_one.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["dropped"], df["cindex_oof"], color=_palette(len(df), "viridis"))
    plt.ylabel("C-index (OOF)")
    plt.title("Drop-one ablation (light retrain)")
    _rotate_xtick_labels(angle=30, bottom_pad=0.18)
    _safe_savefig(out_dir / "ablation_bar.png")

# ---------------- TEST helpers ----------------
try:
    from lifelines.utils import concordance_index as _cindex_ll
except Exception:
    _cindex_ll = None

def _maybe_to_float_arr(x):
    import numpy as _np
    if hasattr(x, "values"):
        return _np.asarray(x.values, dtype=float)
    return _np.asarray(x, dtype=float)

def _test_branch_report(report_dir: Path, risk_dir: Path, cfg: dict):
    """
    使用 risk_gmoe_strict 下的 risk_fused_test.csv 生成 TEST 集指标和 oof_test.csv。

    Parameters
    ----------
    report_dir : Path
        fusion_report/nsclc_xxx 目录，用于保存输出结果。
    risk_dir : Path
        risk_gmoe_strict 目录，里面应包含 risk_fused_test.csv。
    cfg : dict
        配置字典，目前主要是为了保持接口一致，函数内部并不强依赖。
    """
    import pandas as pd
    from lifelines.utils import concordance_index

    risk_file = risk_dir / "risk_fused_test.csv"
    if not risk_file.exists():
        print(f"[TEST] skip: {risk_file} not found")
        return

    rf = pd.read_csv(risk_file)

    # 只取 test 行；如果没有 split 列，就认为全是 test
    if "split" in rf.columns:
        te = rf.loc[rf["split"] == "test"].copy()
    else:
        te = rf.copy()
        te["split"] = "test"

    if te.empty:
        print("[TEST] skip: no test rows in risk_fused_test.csv")
        return

    # 构造和 oof.csv 类似的列：基本列 + gate 权重列 + 各模态 risk
    base_cols = [c for c in ["pid", "time", "event", "risk", "fold", "split"] if c in te.columns]
    gate_cols = [c for c in te.columns if c.startswith("w_")]
    other_cols = [c for c in te.columns if c.startswith("risk_") and c != "risk"]
    cols_keep = base_cols + gate_cols + other_cols
    te = te[cols_keep]

    # 计算简单的整体指标
    try:
        c_test = float(concordance_index(te["time"], -te["risk"], te["event"]))
    except Exception as e:
        print(f"[TEST] C-index failed: {e}")
        c_test = float("nan")

    try:
        from sklearn.metrics import brier_score_loss
        brier_test = float(brier_score_loss(te["event"], te["risk"]))
    except Exception as e:
        print(f"[TEST] Brier failed: {e}")
        brier_test = float("nan")

    metrics = {
        "c_index_test": c_test,
        "brier_test": brier_test,
        "n_test": int(len(te)),
    }

    # gate 权重的均值 / 方差，帮助你看 TEST 集的模态占比
    if gate_cols:
        w_df = te[gate_cols]
        w_summary = pd.DataFrame({
            "mean_weight": w_df.mean(),
            "std_weight": w_df.std(),
        })
    else:
        w_summary = None

    # 保存结果
    report_dir.mkdir(parents=True, exist_ok=True)

    oof_test_path = report_dir / "oof_test.csv"
    te.to_csv(oof_test_path, index=False)
    print(f"[TEST] oof_test -> {oof_test_path}")

    metrics_path = report_dir / "metrics_test.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\\n")
    print(f"[TEST] metrics -> {metrics_path}")

    if w_summary is not None:
        w_path = report_dir / "gate_weights_test.csv"
        w_summary.to_csv(w_path)
        print(f"[TEST] gate_weights_test -> {w_path}")


    # ---- 2) 计算 test 集各模态平均权重 / 标准差 ----
    if w_cols:
        gate_cols = [c for c in w_cols if c in te.columns]
        if gate_cols:
            gate_summary = (
                te[gate_cols]
                .agg(["mean", "std"])
                .T.reset_index()
                .rename(columns={"index": "feature"})
            )
            gate_summary.to_csv(out_dir / "coef_summary_test.csv", index=False)
            print("[TEST] gate weights summary (mean/std):")
            print(gate_summary)

    # ---- 3) 保持原来的 C-index 计算逻辑 ----
    te["time"]  = te["time"].astype(float)
    te["event"] = te["event"].astype(int)
    te["risk"]  = te["risk"].astype(float)

    if _cindex_ll is not None:
        c_test = _cindex_ll(
            _maybe_to_float_arr(te["time"]),
            -_maybe_to_float_arr(te["risk"]),
            _maybe_to_float_arr(te["event"]).astype(int),
        )
    else:
        c_test = concordance_index(
            te["time"].values,
            -te["risk"].values,
            te["event"].values,
        )
    print(f"[TEST] C-index={c_test:.4f}")


    # 写 metrics.test.json（追加或新建）
    mj = report_dir / "metrics.test.json"
    try:
        d = {"cindex_test": float(c_test)}
        if mj.exists():
            try:
                j = json.loads(mj.read_text(encoding="utf-8"))
                j.update(d)
            except Exception:
                j = d
        else:
            j = d
        mj.write_text(json.dumps(j, indent=2), encoding="utf-8")
    except Exception as e:
        print("[TEST] failed to write metrics.test.json:", e)

    # KM（按中位数二分）
    try:
        km_plot(te, report_dir / "km_test.png", title="KM (Test)")
    except Exception as e:
        print("[TEST] KM failed:", e)

    # 可选：IBS（若 cfg.eval_ibs = True）
    try:
        if bool(cfg.get("eval_ibs", False)):
            oof_file = (report_dir / "risk_fused.csv")
            if not oof_file.exists():
                oof_file = (oof_dir / "risk_fused.csv")
            if oof_file.exists():
                tr = pd.read_csv(oof_file)
                tr = tr.rename(columns={"T": "time", "E": "event"})
                tr = tr[["time", "event", "risk"]].dropna().copy()
                tr["time"]  = tr["time"].astype(float)
                tr["event"] = tr["event"].astype(int)
                tr["risk"]  = tr["risk"].astype(float)
                if "ibs_fold" in globals():
                    ibs, grid, br = ibs_fold(tr, te, m=int(cfg.get("brier_m", 120)))
                    if np.isfinite(ibs):
                        plt.figure(figsize=(8, 5))
                        plt.plot(grid, br)
                        plt.title(f"Test IBS={ibs:.4f}")
                        plt.xlabel("time"); plt.ylabel("Brier")
                        _safe_savefig(report_dir / "test_brier.png")
                else:
                    # 轻量回退：用 train 组的 KM 作为 IPCW 近似
                    from lifelines import KaplanMeierFitter
                    km = KaplanMeierFitter()
                    km.fit(tr["time"].values, event_observed=1 - tr["event"].values)
                    times = np.linspace(np.percentile(tr["time"], 5),
                                        np.percentile(tr["time"], 95),
                                        int(cfg.get("brier_m", 120)))
                    br = []
                    for tt in times:
                        w = float(np.clip(km.predict(max(tt - 1e-7, 0.0)), 1e-8, 1.0))
                        y = (te["time"].values <= tt).astype(float)
                        s = (te["risk"].values <= np.median(te["risk"].values)).astype(float)
                        br.append(np.mean(((y - s) ** 2) / w))
                    ibs = float(np.trapz(br, times) / (times[-1] - times[0] + 1e-8))
                    plt.figure(figsize=(8, 5))
                    plt.plot(times, br)
                    plt.title(f"Test IBS≈{ibs:.4f} (fallback)")
                    plt.xlabel("time"); plt.ylabel("Brier")
                    _safe_savefig(report_dir / "test_brier.png")
    except Exception as e:
        print("[TEST] IBS failed:", e)

# ---------------- main ----------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--verbose", action="store_true")
    args=parser.parse_args()

    import yaml
    C=yaml.safe_load(open(args.cfg,"r",encoding="utf-8")) or {}

    mm_dir   = C.get("mm_dir","outputs/mm_inputs")
    oof_dir  = Path(C.get("oof_dir","outputs/risk_gmoe"))
    out_dir  = Path(C.get("out_dir","outputs/fusion_report_allin1"))
    use_keys = list(C.get("use",[])) or []
    risks    = dict(C.get("risks",{}))
    ensure_dir(out_dir)

    # ★★★ 新增：只画这些 split（默认只画 'val'） ★★★
    keep_splits = [str(x).lower() for x in C.get("oof_split", ["val"])]

    skip_plots  = bool(C.get("skip_plots", False))
    eval_ibs    = bool(C.get("eval_ibs", True))
    bs_cfg      = C.get("bootstrap", {"enabled":True, "n":2000, "seed":2025})

    subs  = load_subjects(mm_dir)
    folds = load_folds(mm_dir)
    base  = subs.merge(folds, on="pid", how="inner")
    print(f"[mm] subjects={len(subs)}  folds rows={len(folds)}  folds uniq={sorted(base['fold'].unique().tolist())}")
    print(f"[cfg] keep oof split -> {keep_splits}")

    # 1) 组装 OOF：优先使用 16/19 的 cv*_val_pred.csv
    val_files = sorted(oof_dir.glob("cv*_val_pred.csv"))
    tr_files  = sorted(oof_dir.glob("cv*_tr_pred.csv"))
    oof_list=[]
    for vf in val_files:
        df=pd.read_csv(vf)
        if "pid" not in df.columns:
            df = df.rename(columns={_pick_id_col(df): "pid"})
        df["pid"] = df["pid"].astype(str).map(_clean_pid)
        ren={}
        for c in df.columns:
            lc = str(c).lower()
            if lc in ("os_time", "time_days"): ren[c] = "time"
            if lc in ("os_event", "status", "event_observed"): ren[c] = "event"
        if ren: df = df.rename(columns=ren)
        assert {"pid", "time", "event", "risk"}.issubset(df.columns), f"{vf} 缺少列"
        m = re.search(r"cv(-?\d+)_val_pred\.csv", vf.name)
        fold = int(m.group(1)) if m else -99
        df["fold"]  = fold
        df["split"] = "val"      # ★★★ 明确 val
        oof_list.append(
            df[["pid","time","event","risk","fold","split"] + [c for c in df.columns if c.startswith("w_")]]
        )

    if oof_list:
        oof = pd.concat(oof_list, axis=0, ignore_index=True)
        print(f"[oof] from cv*_val_pred.csv  rows={len(oof)}")
    else:
        fused_path = oof_dir / "risk_fused.csv"
        if fused_path.exists():
            df=pd.read_csv(fused_path)
            if "pid" not in df.columns: df=df.rename(columns={_pick_id_col(df):"pid"})
            df["pid"]=df["pid"].astype(str).map(_clean_pid)
            ren={}
            for c in df.columns:
                lc=str(c).lower()
                if lc in ("os_time","time_days"): ren[c]="time"
                if lc in ("os_event","status","event_observed"): ren[c]="event"
            if ren: df=df.rename(columns=ren)
            df=df.merge(folds, on="pid", how="left")
            df["split"]=df.get("split","test")
            oof=df[["pid","time","event","risk","fold","split"]+[c for c in df.columns if c.startswith("w_")]].copy()
            print(f"[oof] from risk_fused.csv  rows={len(oof)}")
        else:
            raise SystemExit(f"未找到 {oof_dir}/cv*_val_pred.csv 或 risk_fused.csv")

    # ★★★ 只保留 keep_splits（默认 val）★★★
    if "split" in oof.columns:
        before=len(oof)
        oof=oof[oof["split"].astype(str).str.lower().isin(keep_splits)].copy()
        print(f"[oof] filter by split {keep_splits}: {before} -> {len(oof)} rows")

    oof=_clean_for_metrics(oof, ("time","risk","event"), name="oof")

    c_oof=float(concordance_index(oof["time"], -oof["risk"], oof["event"]))
    print(f"[OOF] C-index={c_oof:.4f}")

    if not skip_plots:
        p_lr=km_plot(oof, out_dir/"km_oof.png", title="KM (OOF)")
    else:
        thr=oof["risk"].median()
        hi=oof[oof["risk"]>=thr]; lo=oof[oof["risk"]<thr]
        p_lr=float(logrank_test(hi["time"], lo["time"], hi["event"].astype(bool), lo["event"].astype(bool)).p_value)
    print(f"[OOF] log-rank p={p_lr:.3g}")

    ibs_oof=float("nan")
    if eval_ibs:
        tr_list=[]
        for tf in tr_files:
            df=pd.read_csv(tf)
            if "pid" not in df.columns: df=df.rename(columns={_pick_id_col(df):"pid"})
            df["pid"]=df["pid"].astype(str).map(_clean_pid)
            ren={}
            for c in df.columns:
                lc=str(c).lower()
                if lc in ("os_time","time_days"): ren[c]="time"
                if lc in ("os_event","status","event_observed"): ren[c]="event"
            if ren: df=df.rename(columns=ren)
            m=re.search(r"cv(\d+)_tr_pred\.csv", tf.name); fold=int(m.group(1)) if m else -1
            df["fold"]=fold; df["split"]="train"
            tr_list.append(df[["pid","time","event","risk","fold","split"]])
        if tr_list and "fold" in oof.columns:
            tr_all=pd.concat(tr_list,axis=0,ignore_index=True)
            rows=[]
            plt.figure(figsize=(8.5,5.0))
            for k in sorted(oof["fold"].dropna().unique().astype(int)):
                tr=tr_all[tr_all["fold"]==k].copy()
                va=oof[oof["fold"]==k].copy()
                tr=_clean_for_metrics(tr, ("time","risk","event"), name=f"tr@fold{k}")
                va=_clean_for_metrics(va, ("time","risk","event"), name=f"va@fold{k}")
                ibs, grid, brier = ibs_fold(tr, va)
                if np.isfinite(ibs):
                    rows.append({"fold":k, "ibs":ibs})
                    plt.plot(grid, brier, alpha=0.8, lw=2, label=f"fold{k} (IBS={ibs:.3f})")
            if rows:
                ibs_df=pd.DataFrame(rows); ibs_oof=float(ibs_df["ibs"].mean())
                ibs_df.to_csv(out_dir/"ibs_summary.csv", index=False)
                plt.xlabel("Time"); plt.ylabel("Brier score"); plt.legend()
                plt.title("OOF Brier (per-fold)")
                _safe_savefig(out_dir/"oof_brier.png")

    bs_ci=None
    if bool(bs_cfg.get("enabled",True)):
        bs_ci=bootstrap_cindex(oof, out_dir, n=int(bs_cfg.get("n",2000)), seed=int(bs_cfg.get("seed",2025)))
        print(f"[Bootstrap] C-index 2.5%/50%/97.5% = {bs_ci}")

    use_w=[k for k in use_keys if f"w_{k}" in oof.columns]
    if use_w:
        plot_coef_summary(oof, use_w, out_dir)

    if risks:
        cnt=[]
        for k,p in risks.items():
            try:
                df=read_risk_csv(p,k); avail=df[f"risk_{k}"].notna().sum()
                cnt.append((k, int(avail)))
            except Exception as e:
                print(f"[coverage] {k}: read fail -> {e}")
        if cnt:
            names=[c[0] for c in cnt]; vals=[c[1] for c in cnt]
            plt.figure(figsize=(7.2,4.2))
            plt.bar(names, vals, color=_palette(len(vals),"viridis"))
            plt.title("Coverage by modality"); plt.ylabel("#available"); plt.ylim(0,len(subs))
            _rotate_xtick_labels(angle=30, bottom_pad=0.18)
            _safe_savefig(out_dir/"coverage_bar.png")

    if bool(C.get("do_ablation", False)) and risks and len(use_keys)>=2:
        base2 = base.copy()
        base2["fold"]  = pd.to_numeric(base2["fold"], errors="coerce").astype("Int64")
        base2["time"]  = pd.to_numeric(base2["time"], errors="coerce")
        base2["event"] = pd.to_numeric(base2["event"], errors="coerce")

        for k in use_keys:
            p=risks.get(k)
            if not p:
                print(f"[coverage] {k}: no path in cfg.risks, skip")
                continue
            r=read_risk_csv(p,k)
            cols = ["pid", f"risk_{k}"] + (["fold"] if "fold" in r.columns else [])
            r = r[cols].copy()
            if "fold" in r.columns:
                r["fold"] = pd.to_numeric(r["fold"], errors="coerce").astype("Int64")
                base2 = base2.merge(r, on=["pid","fold"], how="left")
            else:
                base2 = base2.merge(r, on="pid", how="left")
            avail = base2[f"risk_{k}"].notna().sum()
            print(f"[coverage] {k}: available={avail}/{len(base2)}")

        base2 = base2.dropna(subset=["time","event"]).copy()
        has_any = any(base2.get(f"risk_{k}", pd.Series([np.nan]*len(base2))).notna().any() for k in use_keys)
        if has_any:
            print("[ablation] start light retrain & drop-one ...")
            light_train_ablation(base2, use_keys, C.get("ablation_train", {}), out_dir)
        else:
            print("[ablation] 无可用风险特征，跳过。")

    metrics = {
        "cindex_oof": float(c_oof),
        "logrank_p":  float(p_lr),
        "ibs_oof":    float(ibs_oof) if np.isfinite(ibs_oof) else None,
        "bootstrap":  {"enabled": bool(bs_cfg.get("enabled",True)),
                       "n": int(bs_cfg.get("n",2000)),
                       "seed": int(bs_cfg.get("seed",2025)),
                       "ci": [float(x) for x in (bs_ci.tolist() if bs_ci is not None else [])]}
    }
    pd.DataFrame(oof).to_csv(out_dir/"oof.csv", index=False)
    json.dump(metrics, open(out_dir/"metrics.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    # ===== TEST 分支（如存在 risk_fused_test.csv 则评估/作图）=====
    try:
        _test_branch_report(out_dir, oof_dir, C)
    except Exception as _e:
        print("[TEST] report failed:", _e)

    with open(out_dir/"README.txt","w",encoding="utf-8") as f:
        f.write(
f"""Fusion report (risk-level)  — OOF only (+ optional TEST)
=======================================
keep split: {keep_splits}

OOF C-index: {metrics['cindex_oof']:.4f}
OOF Logrank: p={metrics['logrank_p']:.3g}
OOF IBS    : {ibs_oof if np.isfinite(ibs_oof) else 'NA'}

Artifacts:
- metrics.json, oof.csv
- km_oof.png
- bootstrap_cindex.csv / bootstrap_hist.png
- ibs_summary.csv / oof_brier.png    (若找到 cv*_tr_pred.csv)
- coef_summary.csv / coef_bar.png, gate_box.png / gate_bar.png  (若 OOF 含 w_* 列)
- coverage_bar.png                    (若 cfg.risks 提供了路径)
- ablation_drop_one.csv / ablation_bar.png  (若 cfg.do_ablation=True 且提供 risks)
- metrics.test.json, km_test.png, test_brier.png  (若找到 risk_fused_test.csv)
""")
    print(f"[save] all artifacts -> {out_dir}")
    print("Done.")

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()
