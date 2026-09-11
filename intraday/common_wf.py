"""共用模块：数据加载 + ATR + 不重叠采样（train.py / test.py 共用，保证口径一致）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_orderflow_v2 import _agg_minute, build_features

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
F = 0.01
COST_BPS = 3          # 单边成本 bps（成交额 × 万三）
SPEED = "/tmp/dummy"  # 占位（未用）


def load_minute():
    """分天流式加载 mbo 数据聚合到分钟级（防 OOM）。"""
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


def build_nonoap(min_df, H, price_only=False):
    """不重叠采样：label 窗口 H 分钟，每 H 分钟采一个。返回 (X, y, e_idx, x_idx)。

    price_only=True 时剔除 4 个订单流特征（ofi/flow_large/nb/avg_size），只用价格/量特征——
    用于对比「订单流特征贡献多少 edge」（实盘中订单流需 Databento 实时订阅，价格免费可得）。
    """
    feats = build_features(min_df, 60)
    if price_only:
        feats = feats.drop(columns=["ofi", "flow_large", "nb", "avg_size"], errors="ignore")
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


def window_slices(n_samples, H, train_days, test_days):
    """生成滚动窗口的 (train_slice, test_slice, window_id) 列表。与原 walk_forward_r 完全一致。"""
    spd = 390 // H
    train_n, test_n = train_days * spd, test_days * spd
    out = []
    start = 0
    wid = 0
    while start + train_n + test_n <= n_samples:
        out.append((slice(start, start + train_n), slice(start + train_n, start + train_n + test_n), wid))
        start += test_n
        wid += 1
    return out
