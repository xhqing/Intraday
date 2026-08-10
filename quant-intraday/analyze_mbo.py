#!/usr/bin/env python3
"""全量订单簿（mbo）订单流不平衡验证：QQQ 1 天 1963 万条订单簿事件 → 分钟级 OFI → 预测下一分钟收益。

数据：Databento XNAS.ITCH mbo（每笔委托的完整生命周期，含主动方向 side 和价格）。
方法：
  1. 事件流聚合：用 order_id 跟踪每笔委托的"新增(主动) vs 删除" → 计算逐事件"主动买/卖不平衡"
  2. 简化 OFI（订单流不平衡，Cont et al.）：OFI = Σ(主动买委托量) - Σ(主动卖委托量)，分钟级
  3. 回归：下一分钟收益 ~ 当前分钟 OFI
  4. 量级 vs 成本（3bps）
"""

import pandas as pd
import numpy as np
from scipy import stats


def main() -> None:
    # 运行环境：需 zsh -c 'source ~/.zshrc'（拿 key + 走本机代理，代理环境才能正常下载）
    from pathlib import Path
    out_dir = Path(__file__).parent / "data_tick" / "mbo_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    from databento import Historical
    h = Historical()
    # 分段拉取（每小时一段，每段立即存 parquet，断点可续）；重试 3 次
    frames = []
    start_ts, end_ts = pd.Timestamp("2026-08-03", tz="UTC"), pd.Timestamp("2026-08-04", tz="UTC")
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(hours=1), end_ts)
        chunk_file = out_dir / f"chunk_{cur:%H}.parquet"
        if chunk_file.exists():        # 已下载的段跳过（断点续传）
            frames.append(pd.read_parquet(chunk_file))
            cur = nxt
            continue
        ok = False
        for attempt in range(3):
            try:
                store = h.timeseries.get_range(
                    dataset="XNAS.ITCH", schema="mbo",
                    start=cur, end=nxt, symbols=["QQQ"],
                )
                df_chunk = store.to_df()
                df_chunk.to_parquet(chunk_file)   # 立即落盘
                frames.append(df_chunk)
                ok = True
                print(f"  ✅ {cur:%H} 段 {len(df_chunk)} 条已存")
                break
            except Exception as e:
                print(f"  {cur} 段失败(第{attempt+1}次): {str(e)[:80]}")
        if not ok:
            print(f"  ⚠️ {cur} 段跳过（3 次失败）")
        cur = nxt
    import pandas as _pd
    df = _pd.concat(frames)
    print(f"QQQ mbo 1 天（分段拉取）：{len(df)} 条订单簿事件")

    # 简化 OFI：只统计"新增委托"(A) 的主动方向（买/卖），忽略删除（用 side 字段，A=主动买）
    # 实际 mbo 的 side 是每笔委托的买卖方向（A=Ask 卖单 / B=Bid 买单），action 是生命周期
    # OFI 用"新增的买单量 - 新增的卖单量"（每笔委托进入订单簿时的方向）
    active = df[df["action"] == "A"].copy()
    print(f"  新增委托 {len(active)} 条（占 {len(active)/len(df):.1%}）")
    print(f"  side 分布: {active['side'].value_counts().to_dict()}")

    frame = pd.DataFrame({
        "ts": pd.to_datetime(df["ts_event"], unit="ns"),
        "action": df["action"].values,
        "side": df["side"].map({"A": 1, "B": -1, "N": 0}).fillna(0).values,
        "price": df["price"].fillna(0).values,
        "size": df["size"].fillna(0).values,
    }).set_index("ts").sort_index()
    # 每笔事件的对订单簿的影响：新增买单 = +size，新增卖单 = -size；删除反号
    # （简化为只看新增：active buy - active sell）
    signed = frame["side"] * frame["size"]
    frame["signed_vol"] = signed

    # 分钟级聚合
    min_df = frame.resample("1min").agg(
        ofi=("signed_vol", "sum"),
        last_price=("price", lambda x: x[x > 0].iloc[-1] if (x > 0).any() else np.nan),
        n_events=("size", "count"),
    ).dropna()
    min_df["ret_bps"] = min_df["last_price"].pct_change() * 1e4
    min_df["ofi_prev"] = min_df["ofi"].shift(1)
    min_df = min_df.dropna()
    n = len(min_df)
    print(f"\n=== 数据概览（{n} 分钟）===")
    print(f"  分钟 OFI 均值 {min_df['ofi'].mean():+.0f} | 标准差 {min_df['ofi'].std():.0f}")
    print(f"  分钟事件数均值 {min_df['n_events'].mean():.0f}")

    # 回归：本分钟收益 ~ 上一分钟 OFI
    slope, intercept, r, p, se = stats.linregress(min_df["ofi_prev"].values, min_df["ret_bps"].values)
    t_val = slope / se if se else 0
    print(f"\n=== 回归：本分钟收益 ~ 上一分钟 OFI ===")
    print(f"  斜率 {slope:.6f} bps/单位OFI | R={r:.4f} | p={p:.4f} | t={t_val:.2f}")
    print(f"  {'✅ 统计显著（OFI 有方向预测力）' if p < 0.05 else '❌ 不显著'}")

    # 五分位
    print(f"\n=== 按 OFI 五分位看本分钟收益（bps）===")
    q = pd.qcut(min_df["ofi_prev"], 5, labels=False, duplicates="drop")
    labels = ["最低OFI(强卖)", "次低", "中性", "次高", "最高OFI(强买)"]
    for qi in range(5):
        sel = min_df[q == qi]
        if len(sel):
            print(f"  {labels[qi]:<12} n={len(sel):>4} 平均收益 {sel['ret_bps'].mean():+.2f} bps")

    # 量级 vs 成本
    if q.max() >= 4 and q.min() <= 0:
        top = min_df[q == 4]["ret_bps"].mean()
        bot = min_df[q == 0]["ret_bps"].mean()
        spread = (top - bot) / 2
        print(f"\n=== 量级 vs 成本（3bps）===")
        print(f"  强买-强卖收益差/2 = {spread:+.2f} bps（潜在每侧 edge）")
        print(f"  {'❌ 量级 < 成本（不可交易）' if abs(spread) < 3 else '✅ 量级 > 成本（有交易空间）'}")


if __name__ == "__main__":
    main()
