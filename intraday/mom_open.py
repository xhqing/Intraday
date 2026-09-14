#!/usr/bin/env python3
"""开盘动量检验：首半小时收益是否预测当日剩余 / 末半小时（事件驱动线实验 1，2026-09-13）。

背景：订单流方向研究收官后转向事件驱动线。逻辑缺口：F-label 检验（AUC 0.50）测的是
全时段平均信号——「信息稀释」假设在信息集中释放的时刻最弱。开盘是每日信息最密集的
窗口（隔夜信息 + 集合竞价），Gao-Han-Li-Zhou (2018, JFE) 文献报告「首半小时收益预测
末半小时」（market intraday momentum）。本实验在富途 QQQ/SPY 6.5 年 1m 数据上复现检验。

口径（美东时间，富途 1m 首根 9:31 的 open ≈ 集合竞价开盘价）：
  r_first = 9:31 open → 10:00 close（首半小时）
  r_mid   = 10:00 → 15:30（当日中段）
  r_last  = 15:30 → 16:00（末半小时；半日（数据不足 390 分钟）跳过）
检验：
  ① IC：corr(r_first, r_mid) / corr(r_first, r_last)（Pearson + Spearman）
  ② 分组：按 r_first 五分位，看 r_mid / r_last 的组均值（单调性）
  ③ 扣成本交易：10:00 按 r_first 方向进场持有到 15:30（交易 r_mid）；
     15:30 按方向进场持有到 16:00（交易 r_last，文献口径）。双边 3bps×2。
  ④ 按年稳定性。

判定：IC 显著 + 分组单调 + 扣成本为正且多数年份为正 → 开盘窗口存在可开采信号。

用法：python3 mom_open.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

QC = Path(__file__).parent.parent / ".claude" / "skills" / "quant" / "data_cache"
COST = 6e-4   # 双边 3bps × 2


def day_slices(df: pd.DataFrame) -> pd.DataFrame:
    """每日取 r_first / r_mid / r_last。"""
    df = df.set_index("time")
    rows = []
    for d, g in df.groupby(df.index.date):
        g = g.sort_index()
        if len(g) < 300:          # 半日 / 数据缺失日跳过
            continue
        t = g.index
        try:
            o1 = g.loc[t[t.hour == 9][0], "open"] if (t.hour == 9).any() else np.nan
        except (KeyError, IndexError):
            continue
        m100 = g[(t.hour == 10) & (t.minute == 0)]
        m1530 = g[(t.hour == 15) & (t.minute == 30)]
        last = g.iloc[-1]
        if len(m100) == 0 or len(m1530) == 0:
            continue
        c100, o1530, c1600 = m100["close"].iloc[0], m1530["open"].iloc[0], last["close"]
        if any(v <= 0 or not np.isfinite(v) for v in (o1, c100, o1530, c1600)):
            continue
        rows.append({
            "date": d,
            "r_first": c100 / o1 - 1,
            "r_mid": o1530 / c100 - 1,
            "r_last": c1600 / o1530 - 1,
        })
    return pd.DataFrame(rows)


def run(sym: str):
    df = pd.read_parquet(QC / sym / "1m.parquet")
    dd = day_slices(df)
    print(f"\n=== {sym} 开盘动量（{len(dd)} 交易日，{dd.date.min()} ~ {dd.date.max()}）===")
    for target in ("r_mid", "r_last"):
        ic_p = dd["r_first"].corr(dd[target])
        ic_s = dd["r_first"].corr(dd[target], method="spearman")
        print(f"  corr(r_first, {target}): Pearson {ic_p:+.4f} | Spearman {ic_s:+.4f}")
        q = pd.qcut(dd["r_first"], 5, labels=False, duplicates="drop")
        grp = dd.groupby(q)[target].agg(["mean", "count"])
        mono = " ".join(f"{v*1e4:+.1f}" for v in grp["mean"])
        print(f"  按 r_first 五分组 {target} 均值(bps): [{mono}]  (Q1→Q5)")
        # 扣成本交易：按 r_first 方向
        sig = np.sign(dd["r_first"])
        net = sig * dd[target] - COST * 1.0
        print(f"  方向交易 {target}（扣双边 6bps）: {net.mean()*1e4:+.2f} bps/日 | "
              f"胜率 {(net>0).mean()*100:.1f}% | t={net.mean()/(net.std()/np.sqrt(len(net))):+.2f}")
        net_y = (dd.assign(y=pd.to_datetime(dd.date).dt.year, net=net)
                   .groupby("y")["net"].agg(["mean", "count"]))
        by_year = " ".join(f"{int(y)}:{m*1e4:+.1f}" for y, m in zip(net_y.index, net_y["mean"]))
        print(f"  分年净 bps: {by_year}")


if __name__ == "__main__":
    for sym in ("US.QQQ", "US.SPY"):
        run(sym)
