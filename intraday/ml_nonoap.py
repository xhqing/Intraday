#!/usr/bin/env python3
"""不重叠采样 + 一次一仓 + 持仓 H 分钟的 ML 实验。

修正（用户指出）：
  1. 不重叠采样：label 窗口 H 分钟，样本每隔 H 分钟采一个（t, t+H, t+2H...），
     消除相邻样本重叠（相邻 label 不再高度相关）。
  2. 一次一仓 + 持仓 H 分钟：信号来开仓，持仓 H 分钟平（吃满盈利窗口）。
  3. 扫描 H=6/8/10/15/20 找最优（AUC + 扣真实成本净收益）。

用法：python3 ml_nonoap.py [--H-list 6 8 10 15 20] [--train-days 8]
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
F = 0.01   # 仓位 1%


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


def build_nonoap_dataset(min_df, H):
    """不重叠采样：label 窗口 H 分钟，样本每隔 H 分钟采一个。
    返回 (X, y, close, idx)，每个样本对应 [t, t+H] 收益方向，样本间不重叠。
    """
    feats = build_features(min_df, 60)
    close = min_df["close"]
    # 不重叠采样点：每 H 分钟一个（从有特征处开始）
    n = len(min_df)
    idxs = list(range(0, n - H, H))   # t = 0, H, 2H, ... 不重叠
    X_list, y_list, close_list, idx_list = [], [], [], []
    for t in idxs:
        if feats.iloc[t].isna().all():
            continue
        future_ret = close.iloc[t + H] / close.iloc[t] - 1
        if not np.isfinite(future_ret):
            continue
        X_list.append(feats.iloc[t].values)
        y_list.append(1 if future_ret > 0 else 0)
        close_list.append(close.iloc[t])
        idx_list.append(t)
    return (np.array(X_list), np.array(y_list), np.array(close_list),
            min_df.index[np.array(idxs)], min_df.index[np.array(idxs) + H])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H-list", type=int, nargs="+", default=[6, 8, 10, 15, 20])
    parser.add_argument("--train-days", type=int, default=8)
    args = parser.parse_args()

    min_df = load_minute()
    print(f"分钟级数据: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)

    for H in args.H_list:
        X, y, close, t_entry, t_exit = build_nonoap_dataset(min_df, H)
        n = len(X)
        split = int(n * 0.7)
        print(f"\n=== H={H}（持仓 {H} 分钟，不重叠采样 {n} 样本）===", flush=True)
        if split < 50 or n - split < 20:
            print(f"  样本不足"); continue
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
        model.fit(X[:split], y[:split])
        proba = model.predict_proba(X[split:])[:, 1]
        auc = roc_auc_score(y[split:], proba) if len(set(y[split:])) > 1 else float("nan")
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        # 一次一仓回测：档1做空 / 档5做多，持仓 H 分钟，真实成本 3bps*(entry+exit)/entry
        trades = []
        pos, entry_price, pos_end = 0, 0.0, -1
        for i in range(len(proba)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                entry_price = close[split + i]
                pos_end = i + H
            elif pos != 0 and i >= pos_end:
                exit_price = close[split + i]
                gross = pos * (exit_price - entry_price) / entry_price
                cost = 3e-4 * (entry_price + exit_price) / entry_price
                trades.append(gross * 1e4 - cost * 1e4)
                pos = 0
        if not trades:
            print(f"  无交易"); continue
        trades = np.array([t for t in trades if np.isfinite(t)])
        if len(trades) == 0:
            print(f"  无有效交易"); continue
        net_mean = trades.mean()
        win = (trades > 0).mean()
        # g
        ret_r = trades / 1e4   # bps → 收益率
        valid_g = [r for r in ret_r if np.isfinite(r) and 1 + F * r > 0]
        g = sum(math.log(1 + F * r) for r in valid_g) / len(valid_g) if valid_g else 0
        print(f"  AUC {auc:.3f} | 净收益 {net_mean:+.2f} bps/笔 | 胜率 {win:.1%} | "
              f"交易 {len(trades)} | g@1% {g*100:+.3f}%")


if __name__ == "__main__":
    main()
