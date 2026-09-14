#!/usr/bin/env python3
"""成交价 label 终极检验：委托流特征对「真实价格方向」是否有预测力（回测方法改进实验 7）。

背景（2026-09-13 队列模拟 + 延迟曲线之后的根因发现）：
  - 全管道的分钟 close（含 label 锚点 close_t）是 A 委托价——信号分钟最后一张新增委托的
    价格，其中 24% 偏离同分钟最后成交价 >10bps、16% >20bps（深价委托）。
  - 伪影方向性实测：做多信号 close_t 均值低于市场成交价 −28.8bps、做空 +31.0bps——
    模型学到的「方向」主体是委托价偏离的机械回归（锚在市场下方，未来涨回），
    不是真实价格方向。
  - 真实成交价世界（F 进 F 出）毛期望 +0.2bps ≈ 0——旧信号在真实世界无预测力。
  - 富途证伪（成交价特征 + 成交价 label，5.5 年 −6bps）没回答过的问题是：
    **mbo 委托流特征 + 真实成交价 label** 是否仍有预测力（信息源与 label 分离）。

设计（与 verify_2025 完全同口径，仅 label 与评估价改 F 成交价）：
  - 特征：原 15 个不变（委托流 4 + 委托价 OHLCV 派生 11——检验「委托流信息」本身）
  - label：y = F_close[t+H] > F_close[t]（每分钟最后一笔 F 成交价，H=6）
  - 评估：qcut 五档取两端，进场 = F_close[t]（信号分钟最后成交价，≈零延迟市价），
    出场 = F_close[t+H]，双边 6bps——全真实价格口径，无委托价伪影空间

判定：AUC 显著 >0.52 且 F 口径净期望 >0 → 委托流信息真实存在，回执行问题；
      AUC ≈0.5 → 委托流对真实价格无预测力，mbo 方向终结、归档负面结论。

用法：
  python3 verify_flabel.py --year 2025
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, H, TRAIN_DAYS, TEST_DAYS, PARAMS
from ml_orderflow_v2 import build_features

TMP = Path(__file__).parent.parent / "tmp"
MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"


def load_f_minute_year(year: str) -> pd.DataFrame:
    """每日 F 成交价的分钟聚合（每分钟最后一笔成交价），返回按分钟的 F_close。"""
    import pyarrow.parquet as pq

    frames = []
    for day_dir in sorted(MBODIR.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith(year):
            continue
        parts = []
        for f in sorted(day_dir.glob("*.parquet")):
            if f.stat().st_size <= 1024:
                continue
            try:
                t = pq.read_table(f, columns=["action", "price", "ts_event"]).to_pandas()
                parts.append(t[t["action"] == "F"])
            except Exception:
                continue
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["ts"] = df["ts_event"].astype("int64")
        df = df.sort_values("ts")
        df["minute"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.floor("1min")
        fc = df.groupby("minute")["price"].last()
        frames.append(fc.rename("f_close"))
    return pd.concat(frames).sort_index().to_frame()


def run_year(year: str) -> None:
    min_df = add_atr(load_minute_year(year))          # 委托口径分钟（特征用）
    fmin = load_f_minute_year(year)                   # 成交价分钟（label / 评估用）
    print(f"{year}: 委托分钟 {len(min_df)} | 成交分钟 {len(fmin)}", flush=True)

    # 对齐：F_close ffill 到委托分钟索引（缺成交的分钟沿用上一笔成交价）
    f_close = fmin["f_close"].reindex(min_df.index).ffill()
    n_miss = int(f_close.isna().sum())
    valid = f_close.notna().values
    print(f"F_close 对齐：缺失 {n_miss} 分钟（盘前无成交，样本剔除）", flush=True)

    feats = build_features(min_df, 60)
    close_f = f_close.values
    n = len(min_df)

    # 不重叠采样（与 build_nonoap 同款，但 label 用 F 成交价）
    X_list, y_list, e_idx, x_idx = [], [], [], []
    for t in range(0, n - H, H):
        if feats.iloc[t].isna().all() or not valid[t] or not valid[t + H]:
            continue
        fr = close_f[t + H] / close_f[t] - 1
        if not np.isfinite(fr):
            continue
        X_list.append(feats.iloc[t].values)
        y_list.append(1 if fr > 0 else 0)
        e_idx.append(t)
        x_idx.append(t + H)
    X, y = np.array(X_list), np.array(y_list)
    e_idx, x_idx = np.array(e_idx), np.array(x_idx)
    print(f"样本 {len(X)}（F label 正类占比 {y.mean():.3f}）", flush=True)

    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    aucs, nets = [], []
    n_tr = 0
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        # 逐信号评估（一次一仓对齐 verify_2025：信号天然不重叠）
        trades = []
        te_e, te_x = e_idx[te], x_idx[te]
        pos, pos_end, t0 = 0, -1, 0
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                pos_end = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end:
                e_p, x_p = close_f[t0], close_f[pos_end]
                trades.append(pos * (x_p - e_p) / e_p - 3e-4 * (e_p + x_p) / e_p)
                pos = 0
        n_tr += len(trades)
        if trades:
            nets.append(np.mean(trades))
    aucs_f = [a for a in aucs if np.isfinite(a)]
    mean_net = np.mean(nets) if nets else float("nan")
    print(f"\n=== {year} F-label 检验（特征=委托流 15 个，label/评估=F 成交价）===")
    print(f"  AUC: 均值 {np.mean(aucs_f):.4f} | 正窗口(>0.52) {sum(1 for a in aucs_f if a>0.52)}/{len(aucs_f)}")
    print(f"  净收益: {mean_net*1e4:+.1f} bps/笔（F 进 F 出，双边 6bps）| 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)} | 交易 {n_tr}")
    if np.mean(aucs_f) > 0.52 and mean_net > 0:
        print("  ✅ 委托流信息对真实价格有预测力——回到执行问题")
    elif np.mean(aucs_f) > 0.52:
        print("  ⚠️ 方向有预测力但扣成本不正——弱信号")
    else:
        print("  ❌ 委托流特征对真实价格方向无预测力——旧 edge 为委托价伪影，mbo 方向终结")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], default="2025")
    args = parser.parse_args()
    run_year(args.year)
