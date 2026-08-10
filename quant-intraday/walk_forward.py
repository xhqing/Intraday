#!/usr/bin/env python3
"""ML walk-forward 验证：滚动窗口验证「订单流+OHLCV ML 预测 5 分钟做空信号」的样本外稳健性。

方法（消除参数选择泄漏 + 验证跨时段稳健）：
  1. 数据：全部 mbo 天（现有 14 天 + 更多拉取中），聚合分钟级（分天流式防 OOM）
  2. 特征/目标同 ml_orderflow_v2（订单流 + OHLCV，预测 H=5 分钟收益方向）
  3. 滚动窗口：train 固定 N 天 → test 固定 M 天，步长 = M 天（test 无重叠）
  4. 每窗口：只 train 训练，test 验证 AUC + 做空分档（档1）扣成本收益
  5. 判定：多数窗口 AUC>0.52 且做空收益多数正 = 信号稳健

用法：python3 walk_forward.py [--train-days 10] [--test-days 3] [--H 5]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
sys_path = str(Path(__file__).resolve().parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from ml_orderflow_v2 import _agg_minute, build_features   # 复用加载/特征


def load_minute():
    """分天流式加载聚合全部 mbo（防 OOM）。"""
    frames = []
    for day_dir in sorted(MBODIR.iterdir()):
        if not day_dir.is_dir():
            continue
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        parts = [pd.read_parquet(f) for f in files]
        valid_parts = [p for p in parts if len(p)]
        if not valid_parts:   # 整天空（周末/缺数据）→ 跳过，不报错
            continue
        df = pd.concat(valid_parts)
        if len(df):
            frames.append(_agg_minute(df))
        del parts, df
    return pd.concat(frames).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", type=int, default=10)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--n-est", type=int, default=300)
    args = parser.parse_args()

    min_df = load_minute()
    print(f"分钟级数据: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    feats = build_features(min_df, 60)
    future_ret = min_df["close"].shift(-args.H) / min_df["close"] - 1
    y = (future_ret > 0).astype(int)
    ret_bps = future_ret * 1e4
    X = feats.values
    valid = (~feats.isna().all(axis=1)).values & np.isfinite(y.values) & np.isfinite(ret_bps.values)
    X, y, ret_bps = X[valid], y.values[valid], ret_bps.values[valid]
    print(f"有效样本: {len(X)} 分钟，正类 {y.mean():.1%}", flush=True)

    # 按天切窗口（样本按时间排序，分钟/天换算）
    min_per_day = 390   # 美股交易日约 390 分钟
    train_n, test_n = args.train_days * min_per_day, args.test_days * min_per_day
    results = []
    start = 0
    while start + train_n + test_n <= len(X):
        tr = slice(start, start + train_n)
        te = slice(start + train_n, start + train_n + test_n)
        model = lgb.LGBMClassifier(n_estimators=args.n_est, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        # 做空分档（最低 20% 预测概率 → 做空），扣 6bps 成本
        q = pd.qcut(proba, 5, labels=False, duplicates="drop")
        if q.max() >= 1 and (q == 0).any():
            short_ret = (-ret_bps[te][q == 0]).mean() - 6
        else:
            short_ret = float("nan")
        # 做多分档（最高 20% 预测概率 → 做多），扣 6bps 成本
        if q.max() >= 4 and (q == 4).any():
            long_ret = (ret_bps[te][q == 4]).mean() - 6
        else:
            long_ret = float("nan")
        results.append((start // min_per_day, auc, short_ret, long_ret, test_n))
        print(f"窗口{len(results)}: train~day{start//min_per_day}  AUC {auc:.3f}  做空 {short_ret:+.1f} bps  做多 {long_ret:+.1f} bps", flush=True)
        start += test_n

    aucs = [r[1] for r in results if np.isfinite(r[1])]
    shorts = [r[2] for r in results if np.isfinite(r[2])]
    longs = [r[3] for r in results if np.isfinite(r[3])]
    print(f"\n=== walk-forward 汇总（{len(results)} 窗口）===")
    print(f"  AUC: 均值 {np.mean(aucs):.3f} | 正窗口(>0.52) {sum(1 for a in aucs if a>0.52)}/{len(aucs)}")
    if shorts:
        print(f"  做空收益: 均值 {np.mean(shorts):+.1f} bps | 正窗口 {sum(1 for s in shorts if s>0)}/{len(shorts)}")
        print(f"  {'✅ 做空信号稳健' if np.mean(shorts) > 0 and sum(1 for s in shorts if s>0) >= len(shorts)*0.6 else '❌ 做空信号不稳定'}")
    if longs:
        print(f"  做多收益: 均值 {np.mean(longs):+.1f} bps | 正窗口 {sum(1 for s in longs if s>0)}/{len(longs)}")
        print(f"  {'✅ 做多信号稳健' if np.mean(longs) > 0 and sum(1 for s in longs if s>0) >= len(longs)*0.6 else '❌ 做多信号不稳定'}")


if __name__ == "__main__":
    main()
