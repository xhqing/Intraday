#!/usr/bin/env python3
"""订单流 + OHLCV 特征 ML（正式版，用 20 天 mbo 数据，严格 train/valid/test）。

修正 v1 的 3 个 bug：
  1. 目标对齐：用分钟级真实价格（mbo 里每笔的 price 就是秒级价格，直接聚合分钟收盘价）
  2. 样本：20 天 → 数万分钟，足够 ML
  3. 严格时序切分：前 14 天 train / 中 3 天 valid / 后 3 天 test

特征（只用 ≤t，防泄漏）：
  订单流：分钟 OFI（买卖不平衡）、大单资金流、买/卖单笔数、均单量、价格冲击
  价格：分钟动量/波动/量能（从 mbo 价格自建分钟 K）
目标：未来 H 分钟收益方向（H=5/15）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"


def _agg_minute(df):
    """把一天（或几段）的委托 df 聚合到分钟级（向量化 + 清洗异常价）。"""
    df = df[df["action"] == "A"].copy()   # 新增委托
    # 清洗：过滤异常价（每日 P1-P99 之外，去掉错单价如 0/199999 污染 close → ret=inf）
    lo, hi = df["price"].quantile(0.01), df["price"].quantile(0.99)
    df = df[(df["price"] >= lo) & (df["price"] <= hi)]
    df["ts"] = pd.to_datetime(df["ts_event"], unit="ns")
    df["side_v"] = df["side"].map({"A": 1, "B": -1, "N": 0}).fillna(0)
    df["signed"] = df["side_v"] * df["size"].fillna(0)
    df = df.assign(ts=df["ts"], signed=df["signed"], side_v=df["side_v"],
                   price=df["price"].fillna(0), size=df["size"].fillna(0))
    df = df.set_index("ts").sort_index()
    g = df.groupby(pd.Grouper(freq="1min"))
    min_df = pd.DataFrame({
        "ofi": g["signed"].sum(),
        "n_buy": (df[df["side_v"] > 0].groupby(pd.Grouper(freq="1min"))["size"].count()).reindex(g.size().index).fillna(0),
        "n_sell": (df[df["side_v"] < 0].groupby(pd.Grouper(freq="1min"))["size"].count()).reindex(g.size().index).fillna(0),
        "avg_size": g["size"].mean(),
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "vol": g["size"].sum(),
        "large_vol": g.apply(lambda x: x.loc[x["size"] > x["size"].quantile(0.75), "signed"].sum()),  # 简化大单
    }).fillna(0)
    min_df["flow_large"] = min_df["large_vol"]
    return min_df.drop(columns=["large_vol"])


def load_all_mbo() -> pd.DataFrame:
    """分天流式加载聚合（避免 7.7GB 全量内存 OOM），逐天聚合后拼接。"""
    frames = []
    parquets = sorted(MBODIR.glob("*.parquet"))
    if parquets:   # mbo_chunks 结构：直接 parquet
        parts = [pd.read_parquet(f) for f in parquets]
        frames.append(_agg_minute(pd.concat([p for p in parts if len(p)])))
    else:          # mbo_days 结构：按天分目录（逐天读 + 聚合 + 释放，防 OOM）
        for day_dir in sorted(MBODIR.iterdir()):
            if not day_dir.is_dir():
                continue
            files = sorted(day_dir.glob("*.parquet"))
            if not files:
                continue
            parts = [pd.read_parquet(f) for f in files]
            df = pd.concat([p for p in parts if len(p)])
            if len(df):
                frames.append(_agg_minute(df))
            del parts, df   # 释放当天内存，防累积 OOM
    return pd.concat(frames).sort_index()


def build_features(min_df, L=60):
    """构造分钟级特征（只用 ≤t）。"""
    out = pd.DataFrame(index=min_df.index)
    close = min_df["close"]
    for k in (5, 15, 30, 60):
        out[f"mom_{k}"] = close / close.shift(k) - 1
        out[f"vol_{k}"] = close.pct_change().rolling(k).std()
    out["vr"] = min_df["vol"] / min_df["vol"].rolling(60).mean()
    out["ofi"] = min_df["ofi"]
    out["flow_large"] = min_df["flow_large"]
    out["nb"] = min_df["n_buy"] / (min_df["n_buy"] + min_df["n_sell"] + 1)
    out["avg_size"] = min_df["avg_size"]
    out["price_impact"] = (close - close.shift(1)) / (min_df["vol"] + 1) * 1e6
    out["hour"] = min_df.index.hour
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=5, help="预测未来 H 分钟（默认 5）")
    parser.add_argument("--L", type=int, default=60, help="特征窗口 L 分钟（默认 60）")
    parser.add_argument("--n-est", type=int, default=300)
    args = parser.parse_args()

    min_df = load_all_mbo()
    print(f"mbo 分钟级数据: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）")
    feats = build_features(min_df, args.L)

    # 目标：未来 H 分钟收益方向（用分钟 close，真实价格）
    future_ret = min_df["close"].shift(-args.H) / min_df["close"] - 1
    y = (future_ret > 0).astype(int)
    ret_bps = (future_ret * 1e4)

    X = feats.values
    valid = (~feats.isna().all(axis=1)).values & (np.isfinite(y.values)) & (np.isfinite(ret_bps.values))
    X, y, ret_bps = X[valid], y.values[valid], ret_bps.values[valid]
    print(f"有效样本: {len(X)} 分钟，正类 {y.mean():.1%}")

    # 严格时序切分：前 70% train / 中 15% valid / 后 15% test（按时间）
    n = len(X)
    tr_end, va_end = int(n * 0.7), int(n * 0.85)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
    X_te, y_te, ret_te = X[va_end:], y[va_end:], ret_bps[va_end:]

    model = lgb.LGBMClassifier(n_estimators=args.n_est, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               n_jobs=2)   # 限制线程数，避免占满 CPU 影响其他进程
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)
    acc = (pred == y_te).mean()
    auc = roc_auc_score(y_te, proba)
    print(f"\n=== 测试集（最后 15% 时间，样本外）===")
    print(f"  准确率 {acc:.3f}（随机 0.500）| AUC {auc:.3f}（0.5=无预测力）")
    print(f"  {'✅ 订单流+价格 ML 有预测力' if auc > 0.54 else '❌ 基本无预测力'}")

    # 分档可交易性
    q = pd.qcut(proba, 5, labels=False, duplicates="drop")
    labels = ["最空", "偏空", "中性", "偏多", "最多"]
    print(f"\n=== 预测分档实际未来收益（H={args.H} 分钟）===")
    for qi in range(5):
        sel = q == qi
        if sel.sum():
            print(f"  档{qi+1}({labels[qi]}) n={sel.sum():>5} 实际涨 {y_te[sel].mean():.1%} 平均收益 {ret_te[sel].mean():+.2f} bps")
    if q.max() >= 3 and q.min() <= 1:
        long_ret = ret_te[q >= 3].mean() - 6
        short_ret = (-ret_te[q <= 1]).mean() - 6
        print(f"\n  策略化（扣 6bps 双边成本）：做多 {long_ret:+.2f} bps | 做空 {short_ret:+.2f} bps | "
              f"{'✅ 覆盖成本' if max(long_ret, short_ret) > 0 else '❌ 无法覆盖成本'}")

    imp = sorted(zip(feats.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n特征重要性 Top5: {[(c, round(v)) for c, v in imp[:5]]}")


if __name__ == "__main__":
    main()
