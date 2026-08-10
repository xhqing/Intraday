#!/usr/bin/env python3
"""资金流（大单分层）验证：从 mbo 全量订单簿按 size 分层，算大单净流入 → 预测下一分钟收益。

理论依据（smart money 假说）：大单（机构）可能比小单（散户）更知情——全量 OFI 里大单小单
方向互相抵消可能不显著，但只统计大单的"资金流"可能显著。本脚本验证这个可能性。

数据：quant-swing/data_tick/mbo_chunks/*.parquet（QQQ mbo 全量订单簿，2026-08-03）。
方法：
  1. 从每笔订单簿事件取新增委托（action=A）的买卖方向（side）和数量（size）
  2. 按 size 分层：大单（> 日均 size 的某分位）vs 小单
  3. 分钟级大单净流入 = Σ(大单买) - Σ(大单卖)
  4. 回归：下一分钟收益 ~ 当前分钟大单净流入 + 量级 vs 成本（3bps）
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHUNKS = Path(__file__).parent / "data_tick" / "mbo_chunks"


def main() -> None:
    # 读取所有已拉取的 mbo 段
    frames = []
    for f in sorted(CHUNKS.glob("*.parquet")):
        df = pd.read_parquet(f)
        frames.append(df)
        print(f"  {f.name}: {len(df)} 条")
    df = pd.concat(frames)
    print(f"总订单簿事件: {len(df)} 条")

    # 只取新增委托（action=A），即委托进入订单簿（新挂单）
    new = df[df["action"] == "A"].copy()
    print(f"新增委托: {len(new)} 条（{len(new)/len(df):.1%}）")
    print(f"side 分布: {new['side'].value_counts().to_dict()}")

    # size 分层：大单阈值 = 单笔 size 的 75 分位（> 该值 = 大单，机构级）
    sizes = new["size"].fillna(0).values
    thr = np.quantile(sizes, 0.75)
    print(f"\n单笔 size 分位: P50={np.quantile(sizes,0.5):.0f} P75={thr:.0f} P90={np.quantile(sizes,0.9):.0f} P99={np.quantile(sizes,0.99):.0f}")
    print(f"大单阈值（>P75）: {thr:.0f} 股")

    # 分层：大单 vs 小单，各自的有符号量（买正卖负）
    new = new.assign(
        side_v=new["side"].map({"A": 1, "B": -1, "N": 0}).fillna(0).values,
        ts=pd.to_datetime(new["ts_event"], unit="ns"),
    )
    new["signed"] = new["side_v"] * new["size"].fillna(0)
    new["is_large"] = new["size"].fillna(0) > thr
    new = new.set_index("ts").sort_index()

    # 分钟级聚合：全量 / 大单 / 小单 的有符号净流入 + 收盘价
    agg = new.resample("1min").agg(
        flow_all=("signed", "sum"),
        flow_large=("signed", lambda x: x[new.loc[x.index, "is_large"]].sum()),
        flow_small=("signed", lambda x: x[~new.loc[x.index, "is_large"]].sum()),
        last_price=("price", lambda x: x[x > 0].iloc[-1] if (x > 0).any() else np.nan),
    ).dropna()
    agg["ret_bps"] = agg["last_price"].pct_change() * 1e4
    agg["large_prev"] = agg["flow_large"].shift(1)
    agg["all_prev"] = agg["flow_all"].shift(1)
    agg = agg.dropna()
    n = len(agg)
    print(f"\n分钟级数据: {n} 分钟")
    print(f"  大单净流入均值 {agg['flow_large'].mean():+.0f} | 标准差 {agg['flow_large'].std():.0f}")

    # 回归：本分钟收益 ~ 上一分钟大单净流入（资金流）
    for col, label in [("all_prev", "全量OFI(对照)"), ("large_prev", "大单资金流")]:
        X = agg[col].values
        y = agg["ret_bps"].values
        slope, intercept, r, p, se = stats.linregress(X, y)
        t_val = slope / se if se else 0
        print(f"\n=== 回归：本分钟收益 ~ 上一分钟{label} ===")
        print(f"  斜率 {slope:.6f} bps/单位 | R={r:.4f} | p={p:.4f} | t={t_val:.2f}")
        print(f"  {'✅ 统计显著（有方向预测力）' if p < 0.05 else '❌ 不显著'}")

    # 大单资金流五分位：看本分钟平均收益
    print(f"\n=== 按大单资金流五分位看本分钟收益（bps）===")
    q = pd.qcut(agg["large_prev"], 5, labels=False, duplicates="drop")
    labels = ["最大净流出(机构卖)", "次低", "中性", "次高", "最大净流入(机构买)"]
    for qi in range(5):
        sel = agg[q == qi]
        if len(sel):
            print(f"  {labels[qi]:<18} n={len(sel):>4} 平均收益 {sel['ret_bps'].mean():+.2f} bps")

    # 量级 vs 成本
    if q.max() >= 4 and q.min() <= 0:
        top = agg[q == 4]["ret_bps"].mean()
        bot = agg[q == 0]["ret_bps"].mean()
        spread = (top - bot) / 2
        print(f"\n=== 量级 vs 成本（3bps）===")
        print(f"  机构买-机构卖收益差/2 = {spread:+.2f} bps（潜在每侧 edge）")
        print(f"  {'❌ 量级 < 成本（不可交易）' if abs(spread) < 3 else '✅ 量级 > 成本（有交易空间）'}")


if __name__ == "__main__":
    main()
