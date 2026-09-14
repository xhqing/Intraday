#!/usr/bin/env python3
"""A 股隔夜反向修正·近年窗口定版：2023-2026 信号还剩多少（A 股线实验 3，2026-09-13）。

背景：执行严谨化（etf_rev.py）显示信号真实（t=2.5~3.5、三标的一致）但分年衰减——
利润集中在 2020-2022，2023-2026 三标的均值仅 +0.3~+3.1bps/笔。按「只有近年算数」
的防衰减纪律，本实验只用近年窗口（2023-01-01 起）回答：执行口径不变的前提下，
近年净期望是否仍有 ≥5bps（活着 → 进正式回测；不足 → 这条线归档）。

设计：
  - 主口径（沿用全样本初筛选定，不在检验窗口内重新选参，防数据窥探）：
    r_us < −1.5% 买入、9:32 进场、持有 30 分钟、双边佣金 2bps + 滑点 2bps
  - 三标的组合：逐笔池化 + 日聚合（同日三标的信号强相关——r_us 同源，日聚合消除
    相关性对 t 值的虚增，是更诚实的口径）
  - 近年内部趋势：2023/2024/2025/2026 分年——检验「衰减是否仍在继续」
  - 敏感性（参考展示，不作选择依据）：持有 30/60 分钟 × 阈值 1.0/1.5/2.0% ×
    进场 9:31/9:32/9:35

用法：python3 etf_recent.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from etf_rev import load_us, build_days

QC = Path(__file__).parent.parent / ".claude" / "skills" / "quant" / "data_cache"
SYMS = ("SH.513050", "SZ.159920", "SH.513100")
RECENT = pd.Timestamp("2023-01-01").date()
COST = 4e-4   # 双边佣金 2bps + 单边滑点 2bps（保守）


def net_series(m: pd.DataFrame, thr=0.015, entry="px_e32", hold="px_30") -> pd.Series:
    sel = m[m["r_us"] < -thr]
    s = sel[hold] / sel[entry] - 1 - COST
    s.index = pd.to_datetime(pd.to_datetime(sel["date"]).dt.date)
    return s


def run():
    us = load_us()
    ms = {s: build_days(s, us) for s in SYMS}

    print("=== 近年窗口定版（2023-01-01 起，主口径：r_us<−1.5% 买入、9:32 进、持 30min、双边 2+2bps）===")
    # 各标的
    for s in SYMS:
        m = ms[s]
        net = net_series(m[m["date"] >= RECENT])
        t = net.mean() / (net.std() / np.sqrt(len(net))) if len(net) > 2 else np.nan
        print(f"  {s}: {net.mean()*1e4:+5.1f} bps/笔 | n={len(net):>3} | t={t:+.2f}")

    # 逐笔池化
    pooled = pd.concat([net_series(m[m["date"] >= RECENT]).rename(s) for s, m in ms.items()], axis=1)
    all_net = pooled.stack()
    t = all_net.mean() / (all_net.std() / np.sqrt(len(all_net)))
    print(f"  逐笔池化: {all_net.mean()*1e4:+5.1f} bps/笔 | n={len(all_net)} | t={t:+.2f}")

    # 日聚合（同日三标的 → 一笔，消除相关性虚增）
    daily = pooled.mean(axis=1)
    daily = daily.dropna()
    t = daily.mean() / (daily.std() / np.sqrt(len(daily)))
    print(f"  日聚合:   {daily.mean()*1e4:+5.1f} bps/日 | n={len(daily)} 日 | t={t:+.2f}")

    # 近年内部趋势
    print("\n  近年分年（日聚合口径）：")
    yr = pd.to_datetime(daily.index).year
    for y in sorted(set(yr)):
        g = daily[yr == y]
        t = g.mean() / (g.std() / np.sqrt(len(g)))
        print(f"    {y}: {g.mean()*1e4:+5.1f} bps/日 | n={len(g):>2} | t={t:+.2f}")

    # 敏感性网格（参考，防窥探声明：不据此选择）
    print("\n  敏感性网格（日聚合近年，均扣双边 2+2bps）：")
    print(f"  {'':>14} | " + " | ".join(f"持{h:>2}min" for h in (30, 60)))
    for thr in (0.010, 0.015, 0.020):
        for entry, name in (("px_e31", "9:31"), ("px_e32", "9:32"), ("px_e35", "9:35")):
            cells = []
            for hold in ("px_30", "px_60"):
                pooled_h = pd.concat(
                    [net_series(m[m["date"] >= RECENT], thr, entry, hold).rename(s) for s, m in ms.items()],
                    axis=1)
                d = pooled_h.mean(axis=1).dropna()
                cells.append(f"{d.mean()*1e4:+5.1f}({len(d):>2})")
            print(f"  thr>{thr:.1%} {name} | " + " | ".join(cells))
    print("  （注：网格含 18 个口径组合，多重比较下 t~2 的格子不足为凭，只看主口径与形态）")


if __name__ == "__main__":
    run()
