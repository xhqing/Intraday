#!/usr/bin/env python3
"""不重叠采样 walk-forward + 止损（R 口径）：一次一仓 + 持仓 H 分钟 + 止损 N×ATR。

与 walk_forward_nonoap 的区别：加止损（R 口径）——
  - 持仓期间：若价格触及止损（做多 low ≤ stop / 做空 high ≥ stop）→ 止损平（R=−1）
  - 否则持仓 H 分钟到点平
  - R = 盈亏 / risk，risk = stop_atr × ATR（止损距离）
对比：无止损（walk_forward_nonoap）vs 有止损（本脚本），看止损是否提升/破坏。

用法：python3 walk_forward_r.py --H 6 --stop-atr 2 --train-days 8 --test-days 2
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


def add_atr(min_df, period=14):
    prev_close = min_df["close"].shift(1)
    tr = pd.concat([
        min_df["high"] - min_df["low"],
        (min_df["high"] - prev_close).abs(),
        (min_df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    min_df["atr"] = tr.rolling(period).mean()
    return min_df


def build_nonoap(min_df, H):
    """不重叠采样：返回 (X, y, entry_idx, exit_idx)。entry_idx 是样本起始分钟，exit_idx 是 H 分钟后。"""
    feats = build_features(min_df, 60)
    close = min_df["close"]
    n = len(min_df)
    X_list, y_list, e_idx, x_idx = [], [], [], []
    for t in range(0, n - H, H):
        if feats.iloc[t].isna().all():
            continue
        fr = close.iloc[t + H] / close.iloc[t] - 1
        if not np.isfinite(fr):
            continue
        X_list.append(feats.iloc[t].values)
        y_list.append(1 if fr > 0 else 0)
        e_idx.append(t)
        x_idx.append(t + H)
    return np.array(X_list), np.array(y_list), np.array(e_idx), np.array(x_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--stop-atr", type=float, default=2.0, help="止损 N×ATR（0=无止损）")
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--n-est", type=int, default=300)
    args = parser.parse_args()
    H, stop_atr = args.H, args.stop_atr

    min_df = load_minute()
    min_df = add_atr(min_df)
    print(f"分钟级: {len(min_df)}（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    n = len(X)
    print(f"不重叠样本: {n}（H={H}）", flush=True)
    close_all = min_df["close"].values
    atr_all = min_df["atr"].values
    high_all = min_df["high"].values
    low_all = min_df["low"].values

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
        # 一次一仓 + 止损：持仓期间逐分钟检查止损（用分钟 low/high），R 口径
        trades = []
        pos, entry_p, risk, stop_p, pos_end_min, pos_entry_i = 0, 0.0, 0.0, 0.0, -1, -1
        te_e, te_x = e_idx[te.start:te.stop], x_idx[te.start:te.stop]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                # 开仓：记录进场分钟 + 退出分钟（开仓样本的 te_x）
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                entry_p = close_all[t0]
                risk = stop_atr * atr_all[t0] if stop_atr > 0 else 1.0
                stop_p = entry_p - pos * risk
                pos_end_min = te_x[i]          # 持仓结束分钟（开仓样本的 H 分钟后）
                pos_entry_i = i                # 开仓样本序号
            elif pos != 0:
                # 持仓中：跳过后续样本（直到平仓），但检查当前持仓是否该平
                # 逐分钟检查止损（从 pos_end_min 前回溯？不——正确：持仓期 = [t0, pos_end_min)）
                exited = False
                # 只在"当前分钟到达 pos_end_min"时处理平仓（持仓结束）
                if te_e[i] >= pos_end_min:
                    # 持仓期间逐分钟检查止损
                    for t in range(t0 + 1, pos_end_min):
                        if pos > 0 and low_all[t] <= stop_p:
                            gross_r = (stop_p - entry_p) / risk
                            trades.append((gross_r, "stop", pos, entry_p, t))
                            exited = True
                            pos = 0
                            break
                        elif pos < 0 and high_all[t] >= stop_p:
                            gross_r = (entry_p - stop_p) / risk
                            trades.append((gross_r, "stop", pos, entry_p, t))
                            exited = True
                            pos = 0
                            break
                    if not exited:
                        # 到点平仓（H 分钟）：R = 盈亏/risk，扣真实成本
                        exit_p = close_all[pos_end_min]
                        gross_r = pos * (exit_p - entry_p) / risk
                        cost_r = 3e-4 * (entry_p + exit_p) / risk
                        trades.append((gross_r - cost_r, "time", pos, entry_p, pos_end_min))
                        pos = 0
        if trades:
            net = np.array([t[0] for t in trades if np.isfinite(t[0])])
            net_mean = net.mean() if len(net) else float("nan")
            win = (net > 0).mean() if len(net) else float("nan")
            n_stop = sum(1 for t in trades if t[1] == "stop")
        else:
            net_mean, win, n_stop = float("nan"), float("nan"), 0
        results.append((start // spd, auc, net_mean, len(trades), n_stop))
        print(f"窗口{len(results)}: day{start//spd}  AUC {auc:.3f}  净R {net_mean:+.3f}  交易{len(trades)} 止损{n_stop}", flush=True)
        start += test_n

    aucs = [r[1] for r in results if np.isfinite(r[1])]
    nets = [r[2] for r in results if np.isfinite(r[2])]
    print(f"\n=== walk-forward 汇总（{len(results)} 窗口，H={H}，止损 {stop_atr}×ATR）===")
    print(f"  AUC: 均值 {np.mean(aucs):.3f} | 正窗口(>0.52) {sum(1 for a in aucs if a>0.52)}/{len(aucs)}")
    if nets:
        print(f"  净R: 均值 {np.mean(nets):+.3f} | 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)}")
        print(f"  {'✅ 稳健' if np.mean(nets) > 0 and sum(1 for s in nets if s>0) >= len(nets)*0.6 else '❌ 不稳定'}")


if __name__ == "__main__":
    main()
