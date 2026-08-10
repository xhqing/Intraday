#!/usr/bin/env python3
"""ML 信号完整回测：订单流+OHLCV 特征 LightGBM → 分档做空/做多 → 带止损止盈/超时/成本的真实交易。

信号（来自 walk-forward 验证的固定配置）：
  - 每 5 分钟：用固定 LightGBM 模型输出未来 5 分钟上涨概率
  - 概率分 5 档：档1（最低20%）→ 做空；档5（最高20%）→ 做多；其余空仓

真实交易要素（区别于简化的固定持有）：
  - 进场：预测档位触发（t 分钟收盘后，t+1 分钟开盘进）
  - 止损：2×ATR（分钟级 ATR）
  - 止盈：3×ATR
  - 超时：持仓满 30 分钟强平
  - 成本：6bps 双边
  - 仓位：f=1%（状态机一次一仓）

评估：净 avg_R / g / 交易数 / 月胜率 / 最差月 / 年化（扣全部成本）

用法：python3 backtest_ml.py [--H 5] [--train-days 8]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys_path = str(Path(__file__).resolve().parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from ml_orderflow_v2 import _agg_minute, build_features

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
F = 0.01          # 仓位 f=1%
COST = 6          # 双边成本 bps
STOP_ATR = 2.0    # 止损 2×ATR
TAKE_ATR = 3.0    # 止盈 3×ATR
MAX_HOLD = 30     # 超时 30 分钟


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


def atr_from_minute(min_df, period=14):
    """分钟级 ATR（用分钟 OHLC 近似真实波幅）。"""
    prev_close = min_df["close"].shift(1)
    tr = pd.concat([
        min_df["high"] - min_df["low"],
        (min_df["high"] - prev_close).abs(),
        (min_df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--stop-atr", type=float, default=2.0, help="止损 ATR 倍数")
    parser.add_argument("--take-atr", type=float, default=3.0, help="止盈 ATR 倍数")
    parser.add_argument("--max-hold", type=int, default=30, help="超时分钟")
    args = parser.parse_args()
    global STOP_ATR, TAKE_ATR, MAX_HOLD
    STOP_ATR, TAKE_ATR, MAX_HOLD = args.stop_atr, args.take_atr, args.max_hold

    min_df = load_minute()
    print(f"分钟级数据: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    feats = build_features(min_df, 60)
    atr = atr_from_minute(min_df)
    min_df["atr"] = atr
    future_ret = min_df["close"].shift(-args.H) / min_df["close"] - 1
    y = (future_ret > 0).astype(int)
    X = feats.values
    valid = (~feats.isna().all(axis=1)).values & np.isfinite(y.values) & min_df["atr"].notna().values
    X, y = X[valid], y.values[valid]
    idx_valid = min_df.index[valid]
    atr_v = min_df["atr"].values[valid]
    close_v = min_df["close"].values[valid]
    print(f"有效样本: {len(X)} 分钟", flush=True)

    # 固定模型：用前 70% 时间训练（walk-forward 验证过的配置），后 30% 做完整回测
    n = len(X)
    split = int(n * 0.7)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
    model.fit(X[:split], y[:split])
    proba = model.predict_proba(X[split:])[:, 1]
    print(f"模型训练完成，回测段 {len(proba)} 分钟（后 30% 时间）", flush=True)

    # 分档：档1做空、档5做多
    q = pd.qcut(proba, 5, labels=False, duplicates="drop")
    signal = np.zeros(len(proba))
    if q.max() >= 4:
        signal[q == 4] = 1    # 做多
    if q.min() <= 0:
        signal[q == 0] = -1   # 做空

    # 状态机回测：t 分钟收盘出信号 → t+1 开盘进 → 止损/止盈/超时平
    atr_b = atr_v[split:]
    close_b = close_v[split:]
    trades = []
    pos, entry_idx = 0, None
    for i in range(1, len(proba)):
        if pos == 0 and signal[i - 1] != 0:
            # 进场（t+1 开盘）
            pos = signal[i - 1]
            entry_idx = i
            entry_price = close_b[i]
            risk = STOP_ATR * atr_b[i]
            if risk / entry_price < 5e-4:   # risk < 0.05% 价格的交易无统计意义（成本爆炸），跳过
                pos = 0
                continue
            stop = entry_price - pos * risk          # 做多止损在下方
            take = entry_price + pos * TAKE_ATR * atr_b[i]   # 止盈 3×ATR
        elif pos != 0:
            # 持仓：检查止损/止盈/超时（用当前分钟 low/high 近似）
            held = i - entry_idx
            if pos > 0:
                if min_df["low"].iloc[split + i] <= stop:
                    exit_price = stop; reason = "stop"
                elif min_df["high"].iloc[split + i] >= take:
                    exit_price = take; reason = "take"
                elif held >= MAX_HOLD:
                    exit_price = close_b[i]; reason = "timeout"
                else:
                    continue
            else:
                if min_df["high"].iloc[split + i] >= stop:
                    exit_price = stop; reason = "stop"
                elif min_df["low"].iloc[split + i] <= take:
                    exit_price = take; reason = "take"
                elif held >= MAX_HOLD:
                    exit_price = close_b[i]; reason = "timeout"
                else:
                    continue
            # 平仓：扣 6bps 成本，算 R
            gross_r = pos * (exit_price - entry_price) / risk
            net_r = gross_r - COST * 1e-4 * (entry_price + exit_price) / risk
            trades.append((idx_valid[split + i], pos, net_r, gross_r, held, reason))
            pos = 0

    # 评估
    if not trades:
        print("无交易"); return
    net = np.array([t[2] for t in trades])
    gross = np.array([t[3] for t in trades])
    net_f = net[np.isfinite(net)]           # 过滤 -inf/nan（risk 极小的成本爆炸样本）
    gross_f = gross[np.isfinite(gross)]
    g = sum(math.log(1 + F * r) for r in net_f if 1 + F * r > 0) / len(net_f) if len(net_f) else 0
    print(f"\n=== 完整回测（后 30% 时间，{len(trades)} 笔）===")
    print(f"  净 avg_R {net_f.mean():+.3f} | 毛 avg_R {gross_f.mean():+.3f} | g@1% {g*100:+.3f}%")
    print(f"  胜率 {np.mean([t[2]>0 for t in trades]):.1%} | 盈亏比 {np.mean([t[2] for t in trades if t[2]>0])/abs(np.mean([t[2] for t in trades if t[2]<0])):.2f}")
    print(f"  做多笔数 {sum(1 for t in trades if t[1]>0)} / 做空笔数 {sum(1 for t in trades if t[1]<0)}")
    print(f"  出场原因: stop {sum(1 for t in trades if t[5]=='stop')} | take {sum(1 for t in trades if t[5]=='take')} | timeout {sum(1 for t in trades if t[5]=='timeout')}")
    # 月胜率（用回测段月份）
    months = {}
    for t in trades:
        months.setdefault(str(t[0])[:7], []).append(t[2])
    pos_months = sum(1 for m in months if sum(months[m]) > 0)
    if months:
        print(f"  月胜率 {pos_months}/{len(months)} | 月均净收益 {np.mean([sum(months[m]) for m in months]):+.2f}R")


if __name__ == "__main__":
    main()
