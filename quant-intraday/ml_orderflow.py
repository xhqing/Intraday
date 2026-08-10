#!/usr/bin/env python3
"""订单流 + OHLCV 特征的 ML 实验：用已拉 mbo 全量订单簿 + 15m K 线，预测未来短周期收益方向。

数据：mbo_chunks/（QQQ mbo 全量订单簿 2026-08-03，约 91 万条）+ data_cache 的 QQQ 15m。
特征（只用到 t 时刻，防泄漏）：
  订单流：分钟级买卖不平衡（OFI）、大单净流入（资金流）、买卖深度差
  价格：动量/波动/量能/均线（15m 前向填充到分钟）
目标：未来 H 分钟收益方向（H=5/15 分钟，订单流高频信号）
验证：1 天数据按时间 70/30 切分（初步可行性，样本少仅作方向判断）
局限：1 天数据不足以定论，仅初步看"订单流特征能否让 ML 学到方向"。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

QUANT = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "quant"
import sys
sys.path.insert(0, str(QUANT))
from data import load_cache

CHUNKS = Path(__file__).parent / "data_tick" / "mbo_chunks"


def load_mbo_minute():
    """加载 mbo 订单簿，聚合到分钟级订单流特征。"""
    frames = []
    for f in sorted(CHUNKS.glob("*.parquet")):
        df = pd.read_parquet(f)
        if len(df):
            frames.append(df)
    df = pd.concat(frames)
    df = df[df["action"] == "A"].copy()   # 只看新增委托（新挂单 = 订单流进入）
    df["ts"] = pd.to_datetime(df["ts_event"], unit="ns")
    df["side_v"] = df["side"].map({"A": 1, "B": -1, "N": 0}).fillna(0)
    df["signed"] = df["side_v"] * df["size"].fillna(0)
    df = df.set_index("ts").sort_index()
    # 大单阈值（P75）
    thr = df["size"].fillna(0).quantile(0.75)
    df["is_large"] = df["size"].fillna(0) > thr
    # 分钟级聚合
    agg = df.resample("1min").agg(
        ofi=("signed", "sum"),                                   # 全量买卖不平衡
        flow_large=("signed", lambda x: x[df.loc[x.index, "is_large"]].sum()),  # 大单资金流
        n_buy=("side_v", lambda x: (x > 0).sum()),
        n_sell=("side_v", lambda x: (x < 0).sum()),
        avg_size=("size", "mean"),
    ).fillna(0)
    return agg


def main() -> None:
    # 1. 订单流分钟特征
    of = load_mbo_minute()
    print(f"订单流分钟特征: {len(of)} 分钟（2026-08-03 交易时段）")
    print(f"  列: {list(of.columns)} | OFI 均值 {of['ofi'].mean():+.0f}")

    # 2. 15m K 线特征（前向填充到分钟）
    df15 = load_cache("US.QQQ", "15m")
    df15 = df15.sort_values("time").reset_index(drop=True)
    close = df15["close"]
    feats15 = pd.DataFrame(index=np.arange(len(df15)))
    for k in (4, 12, 24):
        feats15[f"mom_{k}"] = close / close.shift(k) - 1
        feats15[f"vr_{k}"] = df15["volume"] / df15["volume"].rolling(k).mean()
    feats15["rsi"] = close.pct_change().rolling(14).mean()
    df15_idx = pd.to_datetime(df15["time"])
    feats15.index = df15_idx
    # 前向填充到分钟（每 15 分钟一根，填到分钟）
    feats_min = feats15.resample("1min").ffill()

    # 3. 对齐订单流 + 价格特征
    X = of.join(feats_min, how="inner")
    # 目标：未来 H 分钟收益方向（用分钟级价格，需 QQQ 1 分钟价——这里用 15m 前向填充近似）
    # 简化：目标用未来 H 个 15m 的累计收益（H15 根），近似"未来趋势"
    H15 = 2   # 未来 2 根 15m = 30 分钟
    future = df15["close"].shift(-H15) / df15["close"] - 1
    future_idx = pd.to_datetime(df15["time"])
    future = future.set_axis(future_idx)
    y = (future.resample("1min").ffill() > 0).astype(int)
    # 对齐
    Xy = X.join(pd.DataFrame({"y": y, "ret": future.resample("1min").ffill()}), how="inner").dropna()
    Xv = Xy.drop(columns=["y", "ret"])
    yv = Xy["y"].values
    ret_bps = (Xy["ret"] * 1e4).values
    print(f"\n对齐后样本: {len(Xv)} 分钟，正类占比 {yv.mean():.1%}")

    # 4. 时序切分（1 天数据 70/30）
    n = len(Xv)
    split = int(n * 0.7)
    X_tr, y_tr = Xv.iloc[:split].values, yv[:split]
    X_te, y_te, ret_te = Xv.iloc[split:].values, yv[split:], ret_bps[split:]

    # 5. LightGBM
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)
    acc = (pred == y_te).mean()
    auc = roc_auc_score(y_te, proba)
    print(f"\n=== 测试集（后 30% 分钟，样本外）===")
    print(f"  准确率 {acc:.3f}（随机 0.500）| AUC {auc:.3f}（0.5=无预测力）")
    print(f"  {'✅ 订单流特征有预测力' if auc > 0.55 else '❌ 基本无预测力'}")

    # 6. 分档可交易性
    q = pd.qcut(proba, 5, labels=False, duplicates="drop")
    labels = ["最空", "偏空", "中性", "偏多", "最多"]
    print(f"\n=== 预测分档的实际未来收益（未来 30 分钟）===")
    for qi in range(5):
        sel = q == qi
        if sel.sum():
            print(f"  档{qi+1}({labels[qi]}) n={sel.sum():>4} 实际涨 {y_te[sel].mean():.1%} 平均收益 {ret_te[sel].mean():+.2f} bps")
    long_ret = ret_te[q >= 3].mean() - 6
    short_ret = (-ret_te[q <= 1]).mean() - 6
    print(f"\n  策略化（扣 6bps 双边成本）：做多 {long_ret:+.2f} bps | 做空 {short_ret:+.2f} bps | "
          f"{'✅ 覆盖成本' if max(long_ret, short_ret) > 0 else '❌ 无法覆盖成本'}")

    # 特征重要性（看订单流特征是否被用上）
    imp = sorted(zip(Xv.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n特征重要性 Top5: {[(c, round(v)) for c, v in imp[:5]]}")


if __name__ == "__main__":
    main()
