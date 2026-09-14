#!/usr/bin/env python3
"""深度特征检验：盘口 / 事件流微结构特征对真实价格方向是否有预测力（深挖实验 2）。

背景（2026-09-13）：F-label 终极检验只关闭了原 15 个基础特征（AUC 0.5078/0.5146 ≈ 随机）。
本实验检验 deep_features.py 构建的深度特征家族（CKS 盘口 OFI、多档深度不平衡、
撤单 / 成交流不平衡、事件微结构、mid 动量）——文献有据（Cont-Kukanov-Stoikov 2014 等）
但从未在本数据上测过。label 与评估价全用 F 成交价（伪影检验前置，无委托价锚点）。

特征集（--set 参数）：
  deep   : 深度特征 17 个（spread/depth_imb_1/depth_imb_5/book_vol_5/ofi_cks/add_imb/
           cancel_imb/trade_imb/n_add/n_cancel/cr_ratio/avg_add_size/n_fill/avg_fill_size/
           mid_ret_1/mid_mom_5/mid_mom_15）
  all    : deep + 原 15 个（委托流 + 委托价 OHLCV 派生，对照组）
  orig   : 仅原 15 个（基准，应复现 verify_flabel 的 ~0.51）

设计（与 verify_flabel 完全同款）：H=6 不重叠采样、8 天训练 / 2 天测试滚动、
LightGBM 固定超参、qcut 五档取两端、F 进 F 出双边 6bps。

判定：AUC 显著 >0.52 且净期望 >0 → 深度特征有真实预测力，重启策略研究；
      ≈0.5 → 深度特征家族也关闭（CKS OFI 的预测力若存在也在秒级、分钟尺度已衰减）。

用法：
  python3 verify_deep.py --year 2025 --set deep
  python3 verify_deep.py --year 2025 --set all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, H, TRAIN_DAYS, TEST_DAYS, PARAMS
from verify_flabel import load_f_minute_year
from ml_orderflow_v2 import build_features

TMP = Path(__file__).parent.parent / "tmp"

DEEP_COLS = ["spread_ticks", "depth_imb_1", "depth_imb_5", "book_vol_5", "ofi_cks",
             "add_imb", "cancel_imb", "trade_imb", "n_add", "n_cancel", "cr_ratio",
             "avg_add_size", "n_fill", "avg_fill_size",
             "mid_ret_1", "mid_mom_5", "mid_mom_15"]


def build_matrix(year: str, feat_set: str):
    """返回按分钟网格对齐的 (X, y, f_close)，深度特征 + F label。"""
    deep = pd.read_parquet(TMP / f"deep_feat_{year}.parquet")
    deep = deep[deep["valid"]].copy()
    deep = deep.set_index("ts").sort_index()
    fmin = load_f_minute_year(year)
    f_close = fmin["f_close"].reindex(deep.index).ffill()
    valid = f_close.notna().values
    deep, f_close = deep[valid], f_close[valid]

    X = deep[DEEP_COLS].copy()
    if feat_set in ("all", "orig"):
        min_df = add_atr(load_minute_year(year))
        orig = build_features(min_df, 60).reindex(deep.index)
        for c in orig.columns:
            X[f"o_{c}"] = orig[c]
    if feat_set == "orig":
        X = X[[c for c in X.columns if c.startswith("o_")]]
        X.columns = [c[2:] for c in X.columns]

    # label：F 成交价 6 分钟方向
    fv = f_close.values
    n = len(deep)
    rows, y, idx = [], [], []
    for t in range(0, n - H, H):
        fr = fv[t + H] / fv[t] - 1
        if not np.isfinite(fr) or X.iloc[t].isna().all():
            continue
        rows.append(X.iloc[t].values)
        y.append(1 if fr > 0 else 0)
        idx.append(t)
    return np.array(rows), np.array(y), np.array(idx), fv, deep.index


def run(year: str, feat_set: str) -> None:
    X, y, idx, fv, ts_index = build_matrix(year, feat_set)
    print(f"{year} {feat_set}: 样本 {len(X)}（{X.shape[1]} 特征，正类 {y.mean():.3f}）", flush=True)

    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    aucs, nets = [], []
    n_tr = 0
    imp_acc = np.zeros(X.shape[1])
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        imp_acc += model.feature_importances_
        proba = model.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        trades = []
        te_e, te_x = idx[te], idx[te] + H
        pos, pos_end, t0 = 0, -1, 0
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                pos_end = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end:
                e_p, x_p = fv[t0], fv[pos_end]
                trades.append(pos * (x_p - e_p) / e_p - 3e-4 * (e_p + x_p) / e_p)
                pos = 0
        n_tr += len(trades)
        if trades:
            nets.append(np.mean(trades))

    aucs_f = [a for a in aucs if np.isfinite(a)]
    mean_net = np.mean(nets) if nets else float("nan")
    cols = DEEP_COLS if feat_set == "deep" else None
    print(f"\n=== {year} 深度特征检验（set={feat_set}，F-label，F 进 F 出双边 6bps）===")
    print(f"  AUC: 均值 {np.mean(aucs_f):.4f} | 正窗口(>0.52) {sum(1 for a in aucs_f if a>0.52)}/{len(aucs_f)}")
    print(f"  净收益: {mean_net*1e4:+.1f} bps/笔 | 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)} | 交易 {n_tr}")
    imp = sorted(zip(range(X.shape[1]), imp_acc), key=lambda x: -x[1])[:8]
    print("  特征重要性 Top8:", [(cols[i] if cols else f"f{i}", int(v)) for i, v in imp])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], default="2025")
    parser.add_argument("--set", choices=["deep", "all", "orig"], default="deep")
    args = parser.parse_args()
    run(args.year, args.set)
