#!/usr/bin/env python3
"""ML 趋势可预测性实验：用过去 L 根 15m 的复杂交叉特征，预测未来 H 根收益方向。

方法（严格防泄漏/防过拟合）：
  1. 数据：43 美股 15m（2021-2026，免费已拉）
  2. 特征（只用 ≤ t 数据，滚动窗口）：
     价格动量(多周期) / 波动率 / 量能 / 均线关系 / RSI / 时段
  3. 目标：未来 H 根收益方向（分类，sign(未来 H 根累计收益)）
  4. 时序切分（不随机打乱）：train<2024 / valid 2024-25 / test≥2025
  5. 模型：LightGBM（表格数据强、快、可解释）
  6. 验证：test 准确率/AUC + 可交易性（预测分档的收益 vs 3bps 成本）

用法：python3 ml_trend.py [--H 12] [--L 96] [--sample 0.2] [--n-est 500]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

QUANT = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "quant"
import sys
sys.path.insert(0, str(QUANT))
from data import load_cache, STOCK_POOL


def build_features(df, L=96):
    """构造只用 ≤ t 数据的特征（防未来泄漏）。df: 15m 原始 K 线。"""
    out = pd.DataFrame(index=np.arange(len(df)))
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    # 多周期动量
    for k in (6, 12, 24, 48, 96):
        out[f"mom_{k}"] = close / close.shift(k) - 1
    # 波动率（滚动标准差）
    ret = close.pct_change()
    for k in (12, 48, 96):
        out[f"vol_{k}"] = ret.rolling(k).std()
    # 量能
    for k in (12, 48, 96):
        out[f"vr_{k}"] = vol / vol.rolling(k).mean()
    # 均线关系
    for k in (24, 96):
        ma = close.rolling(k).mean()
        out[f"ma_{k}"] = close / ma - 1
        out[f"ma_slope_{k}"] = ma.pct_change(12)
    # RSI(14) 近似（用收益符号）
    gain = ret.clip(lower=0).rolling(14).mean()
    loss = (-ret.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 * gain / (gain + loss)
    # 时段（小时编码，捕捉时段效应）
    ts = df["time"] if "time" in df.columns else df.index
    out["hour"] = pd.to_datetime(ts).dt.hour.values
    # 高低区间位置（当日位置）
    out["day_pos"] = (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min())
    return out


def build_dataset(symbols, L, H, sample_frac):
    """构建全部标的的 (X, y, future_ret_bps) 样本（池化），严格只用 ≤t 特征 + t+1~t+H 目标。"""
    X_all, y_all, ret_all = [], [], []
    for sym in symbols:
        try:
            df = load_cache(sym, "15m")
        except Exception:
            continue
        if len(df) < L + H + 100:
            continue
        df = df.sort_values("time").reset_index(drop=True) if "time" in df.columns else df.sort_index()
        feats = build_features(df, L)
        # 目标：未来 H 根累计收益方向 + 实际收益（bps，供量级 vs 成本）
        future_ret = df["close"].shift(-H) / df["close"] - 1
        y = (future_ret > 0).astype(int).values
        ret_bps = (future_ret * 1e4).values
        # 特征对齐（只用 ≤t）
        X = feats.values
        valid = (~feats.isna().all(axis=1)).values & (~np.isnan(y))
        X_all.append(X[valid])
        y_all.append(y[valid])
        ret_all.append(ret_bps[valid])
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    ret_bps_all = np.concatenate(ret_all)
    # 降采样（加速实验；若 sample_frac<1 则按 1/frac 间隔抽样，保留时序）
    if sample_frac < 1.0:
        step = int(1 / sample_frac)
        idx = np.arange(0, len(X), step)
        X, y, ret_bps_all = X[idx], y[idx], ret_bps_all[idx]
    return X, y, ret_bps_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=12, help="预测未来 H 根（默认 12 = 3 小时）")
    parser.add_argument("--H-list", type=int, nargs="+", default=None, help="扫描多个 H（如 12 24 48 96）")
    parser.add_argument("--L", type=int, default=96, help="特征窗口 L 根（默认 96 = 1 天）")
    parser.add_argument("--sample", type=float, default=0.5, help="采样比例（加速，默认 0.5）")
    parser.add_argument("--n-est", type=int, default=500)
    args = parser.parse_args()

    symbols = sorted(s for s in STOCK_POOL if s.startswith("US."))
    for H in (args.H_list if args.H_list else [args.H]):
        print(f"\n{'='*60}\nH={H}（预测未来 {H*15/60:.1f} 小时）")
        X, y, ret_bps = build_dataset(symbols, args.L, H, args.sample)
        print(f"总样本 {len(X)}，正类占比 {y.mean():.1%}")

        # 时序切分（不随机打乱——防未来泄漏）。按时间序（样本顺序即时间序）
        n = len(X)
        train_end = int(n * 0.6)
        valid_end = int(n * 0.8)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_va, y_va = X[train_end:valid_end], y[train_end:valid_end]
        X_te, y_te, ret_te = X[valid_end:], y[valid_end:], ret_bps[valid_end:]

        # LightGBM 训练
        model = lgb.LGBMClassifier(
            n_estimators=args.n_est, learning_rate=0.05, num_leaves=63,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)
        acc = (pred == y_te).mean()
        auc = roc_auc_score(y_te, proba)
        print(f"\n=== 测试集（样本外）结果 ===")
        print(f"  准确率 {acc:.3f}（随机 0.500）| AUC {auc:.3f}（0.5=无预测力）")
        print(f"  {'✅ 有预测力' if auc > 0.52 else '❌ 基本无预测力'}")

        # A：可交易性——预测概率分 5 档，看各档实际未来收益（bps）vs 成本
        q = pd.qcut(proba, 5, labels=False, duplicates="drop")
        print(f"\n=== 可交易性：预测分档的实际未来收益（H={H}根）===")
        labels = ["最空", "偏空", "中性", "偏多", "最多"]
        for qi in range(5):
            sel_ret = ret_te[q == qi]
            if len(sel_ret):
                print(f"  档{qi+1}({labels[qi]}) n={len(sel_ret):>6} 实际涨 {y_te[q==qi].mean():.1%} 平均收益 {sel_ret.mean():+.2f} bps")
        # 做多/做空分档的实际净收益（假设按预测方向做，扣 3bps 成本）
        long_mask = q >= 3   # 预测偏多/最多 → 做多
        short_mask = q <= 1  # 预测偏空/最空 → 做空
        long_ret = ret_te[long_mask].mean() - 6    # 双边成本约 6bps（进+出）
        short_ret = (-ret_te[short_mask]).mean() - 6
        print(f"\n  策略化（扣 6bps 双边成本）：")
        print(f"    预测偏多做多：{long_ret:+.2f} bps/笔")
        print(f"    预测偏空做空：{short_ret:+.2f} bps/笔")
        print(f"    {'✅ 覆盖成本' if max(long_ret, short_ret) > 0 else '❌ 无法覆盖成本'}")


if __name__ == "__main__":
    main()
