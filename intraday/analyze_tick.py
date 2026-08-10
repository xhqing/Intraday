#!/usr/bin/env python3
"""订单流不平衡（OFI）初步验证：NVDA 逐笔成交 → 分钟级 OFI → 预测下一分钟收益。

方法：
  1. Databento 拉 NVDA 逐笔成交（trades schema）
  2. 逐笔按 side 聚合到分钟：OFI = Σ(主动买量) - Σ(主动卖量)
  3. 计算分钟收益（分钟收盘价变化，bps）
  4. 回归：本分钟收益 ~ 上一分钟 OFI（t 值/显著性）
  5. 量级 vs 成本（3bps）判断可交易性
"""

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from databento import Historical


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="订单流 OFI 验证")
    parser.add_argument("--symbol", default="NVDA", help="标的（默认 NVDA）")
    parser.add_argument("--schema", default="trades", help="schema（trades 或 mbp1/mbo）")
    parser.add_argument("--days", type=int, default=7, help="拉取天数（默认 7）")
    args = parser.parse_args()
    from datetime import timedelta
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")

    h = Historical()
    store = h.timeseries.get_range(
        dataset="DBEQ.BASIC", schema=args.schema,
        start=start, end=end,
        symbols=[args.symbol],
    )
    df = store.to_df()
    print(f"原始逐笔 {len(df)} 条")

    # side: A=主动买 B=主动卖 N=中性
    frame = pd.DataFrame({
        "ts": pd.to_datetime(df["ts_event"], unit="ns"),
        "side": df["side"].map({"A": 1, "B": -1, "N": 0}).fillna(0).values,
        "size": df["size"].values,
        "price": df["price"].values,
    }).set_index("ts").sort_index()
    frame["signed_vol"] = frame["side"] * frame["size"]

    # 分钟级聚合
    min_df = frame.resample("1min").agg(
        ofi=("signed_vol", "sum"),
        last_price=("price", "last"),
        vol=("size", "sum"),
        n=("size", "count"),
    ).dropna()
    min_df["ret_bps"] = min_df["last_price"].pct_change() * 1e4
    min_df["ofi_prev"] = min_df["ofi"].shift(1)
    min_df = min_df.dropna()
    n = len(min_df)

    print(f"\n=== 数据概览 ===")
    print(f"  {n} 分钟（7/27-8/4，NVDA）")
    print(f"  分钟 OFI 均值 {min_df['ofi'].mean():+.0f} | 标准差 {min_df['ofi'].std():.0f}")
    print(f"  分钟收益均值 {min_df['ret_bps'].mean():+.2f} bps | 标准差 {min_df['ret_bps'].std():.2f}")

    # 回归：本分钟收益 ~ 上一分钟 OFI
    slope, intercept, r, p, se = stats.linregress(min_df["ofi_prev"].values, min_df["ret_bps"].values)
    t_val = slope / se if se else 0
    print(f"\n=== 回归：本分钟收益 ~ 上一分钟 OFI ===")
    print(f"  斜率 {slope:.5f} bps/单位OFI | R={r:.4f} | p={p:.4f} | t={t_val:.2f}")
    print(f"  {'✅ 统计显著（OFI 有方向预测力）' if p < 0.05 else '❌ 不显著'}")

    # 五分位：按上一分钟 OFI 分 5 组，看本分钟平均收益
    print(f"\n=== 按 OFI 五分位看本分钟收益（bps）===")
    q = pd.qcut(min_df["ofi_prev"], 5, labels=False, duplicates="drop")
    labels = ["最低OFI(强卖)", "次低", "中性", "次高", "最高OFI(强买)"]
    for qi in range(5):
        sel = min_df[q == qi]
        if len(sel):
            print(f"  {labels[qi]:<12} n={len(sel):>4} 平均收益 {sel['ret_bps'].mean():+.2f} bps")

    # 量级 vs 成本
    top = min_df[q == 4]["ret_bps"].mean()
    bot = min_df[q == 0]["ret_bps"].mean()
    spread = (top - bot) / 2
    print(f"\n=== 量级 vs 成本（3bps）===")
    print(f"  强买-强卖收益差/2 = {spread:+.2f} bps（潜在每侧 edge）")
    print(f"  {'❌ 量级 < 成本（不可交易）' if abs(spread) < 3 else '✅ 量级 > 成本（有交易空间）'}")


if __name__ == "__main__":
    main()
