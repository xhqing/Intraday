#!/usr/bin/env python3
"""不重叠采样 walk-forward 验证：一次一仓 + 持仓 H 分钟，滚动多窗口确认稳健性。

修正（用户指出）：
  1. 不重叠采样：label 窗口 H 分钟，样本每隔 H 分钟采一个（消除相邻样本重叠）
  2. 一次一仓 + 持仓 H 分钟：信号来开仓，持仓 H 分钟平
  3. 滚动窗口：train 固定 N 天 → test 固定 M 天（不重叠 test），多窗口

用法：python3 walk_forward_nonoap.py --H 6 --train-days 8 --test-days 2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys_path = str(Path(__file__).resolve().parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from ml_orderflow_v2 import _agg_minute, build_features

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
F = 0.01


def load_minute():
    frames = []
    for day_dir in sorted(MBODIR.iterdir()):
        if not day_dir.is_dir():
            continue
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        parts = [pd.read_parquet(f) for f in files]
        valid_parts = [p for p in parts if len(p)]
        if not valid_parts:
            continue
        frames.append(_agg_minute(pd.concat(valid_parts)))
        del parts, valid_parts
    return pd.concat(frames).sort_index()


def build_nonoap(min_df, H):
    """不重叠采样 + 一次一仓所需：返回不重叠样本 (X, y, close, entry_idx, exit_idx)。"""
    feats = build_features(min_df, 60)
    close = min_df["close"]
    n = len(min_df)
    X_list, y_list, close_list, e_idx, x_idx = [], [], [], [], []
    for t in range(0, n - H, H):   # 不重叠采样点
        if feats.iloc[t].isna().all():
            continue
        fr = close.iloc[t + H] / close.iloc[t] - 1
        if not np.isfinite(fr):
            continue
        X_list.append(feats.iloc[t].values)
        y_list.append(1 if fr > 0 else 0)
        close_list.append(close.iloc[t])
        e_idx.append(t)
        x_idx.append(t + H)
    return (np.array(X_list), np.array(y_list), np.array(close_list),
            np.array(e_idx), np.array(x_idx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--n-est", type=int, default=300)
    args = parser.parse_args()
    H = args.H

    min_df = load_minute()
    print(f"分钟级: {len(min_df)}（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    X, y, close, e_idx, x_idx = build_nonoap(min_df, H)
    n = len(X)
    print(f"不重叠样本: {n}（H={H}，每次信号独立）", flush=True)

    # 按天切窗口：min_per_day 用不重叠样本数估算（每天 390 分钟 / H）
    spd = 390 // H
    train_n, test_n = args.train_days * spd, args.test_days * spd
    results = []
    start = 0
    while start + train_n + test_n <= n:
        tr = slice(start, start + train_n)
        te = slice(start + train_n, start + train_n + test_n)
        model = lgb.LGBMClassifier(n_estimators=args.n_est, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        # 一次一仓回测：档1做空/档5做多，持仓 H 分钟，真实成本 3bps*(entry+exit)/entry
        trades = []
        pos, entry_p, pos_end = 0, 0.0, -1
        te_e, te_x = e_idx[te.start:te.stop], x_idx[te.start:te.stop]
        te_close = close[te.start:te.stop]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                entry_p = te_close[i]
                pos_end = i + H
            elif pos != 0 and i >= pos_end:
                exit_p = te_close[i]
                gross = pos * (exit_p - entry_p) / entry_p
                cost = 3e-4 * (entry_p + exit_p) / entry_p
                trades.append(gross * 1e4 - cost * 1e4)
                pos = 0
        if trades:
            trades = np.array([t for t in trades if np.isfinite(t)])
            net_mean = trades.mean() if len(trades) else float("nan")
            win = (trades > 0).mean() if len(trades) else float("nan")
        else:
            net_mean, win = float("nan"), float("nan")
        results.append((start // spd, auc, net_mean, len(trades)))
        print(f"窗口{len(results)}: day{start//spd}  AUC {auc:.3f}  净收益 {net_mean:+.1f} bps  交易 {len(trades)}", flush=True)
        start += test_n

    aucs = [r[1] for r in results if np.isfinite(r[1])]
    nets = [r[2] for r in results if np.isfinite(r[2])]
    print(f"\n=== walk-forward 汇总（{len(results)} 窗口，H={H}）===")
    print(f"  AUC: 均值 {np.mean(aucs):.3f} | 正窗口(>0.52) {sum(1 for a in aucs if a>0.52)}/{len(aucs)}")
    if nets:
        print(f"  净收益: 均值 {np.mean(nets):+.1f} bps | 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)}")
        print(f"  {'✅ 稳健' if np.mean(nets) > 0 and sum(1 for s in nets if s>0) >= len(nets)*0.6 else '❌ 不稳定'}")


if __name__ == "__main__":
    main()
