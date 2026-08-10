#!/usr/bin/env python3
"""数据分析 CLI：回测交易数据的各维度分析。复用 backtest.py 的采集逻辑。

用法（在 quant-swing/ 目录下）：
  python3 analyze.py monthly            # 月收益率分布（描述统计 + 直方图 + 最好/最差月）
  python3 analyze.py hold               # 持仓时间分布（均值/中位/分位 + 直方图）
  python3 analyze.py yearly             # 按年分解（各年池化 g）
  python3 analyze.py symbols            # 各标的 g 排名
  python3 analyze.py worst-month        # 最差月份的实际赔率（R 明细）
  python3 analyze.py gap                # 跳空穿透统计（R<-1 占比/最差）
  python3 analyze.py signal-day         # 信号次日价格行为（开盘 vs VWAP/收盘，回踩触及率）
  python3 analyze.py --scheme 2 ...     # 指定方案（默认 1）
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import common
from backtest import run_backtest, _load, add_indicators

F = 0.02
COST_BPS = 3.0


def collect(symbols, params):
    """收集 (entry_time, sym, direction, R, net_R, hold) 交易列表。"""
    trades = []
    for sym in symbols:
        try:
            df = common.load_day(sym)
        except Exception:
            continue
        if len(df) < 200:
            continue
        r = run_backtest(trend_breakout(), sym, "DAY", params=params, df=df.copy())
        cost = common.MARKET_COST_BPS[common.market_of(sym)]
        for t in r.trades:
            net = t.R - (t.entry_price + t.exit_price) * cost * 1e-4 / t.risk
            trades.append((t.entry_time, t.symbol, t.direction, t.R, net, t.hold_bars))
    return trades


def trend_breakout():
    from strategies import trend_breakout as tb
    return tb


def monthly_returns(trades):
    """f=2% 逐笔复利，按自然月聚合。返回 (rets, months)。"""
    srt = sorted(trades, key=lambda x: x[0])
    W, me = 1.0, {}
    for et, *_ , net_r, _ in srt:
        W *= 1 + F * net_r
        me[et[:7]] = W
    months = sorted(me)
    rets = np.array([me[m] / (me[months[i - 1]] if i > 0 else 1.0) - 1 for i, m in enumerate(months)])
    return rets, months


def cmd_monthly(trades, args):
    rets, months = monthly_returns(trades)
    arr = rets
    n = len(arr)
    print(f"月收益率分布（{n} 个月，f=2% 复利）：")
    print(f"  月均 {arr.mean()*100:+.2f}% | 中位 {np.median(arr)*100:+.2f}% | 标准差 {arr.std()*100:.2f}%")
    print(f"  最差月 {arr.min()*100:+.2f}% | 最好月 {arr.max()*100:+.2f}% | 正月份 {int((arr>=0).sum())}/{n} ({np.mean(arr>=0):.0%})")
    print(f"\n  分布直方图（4% 一档）：")
    bins = np.arange(-24, 25, 4)
    counts, _ = np.histogram(arr * 100, bins=bins)
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i + 1]
        print(f"    [{lo:+3d}%,{hi:+3d}%): {c:>3} 月 ({c/n:>5.1%})")
    print(f"\n  最差 5 个月：")
    for m, r in sorted(zip(months, arr), key=lambda x: x[1])[:5]:
        print(f"    {m}: {r*100:+.2f}%")
    print(f"  最好 5 个月：")
    for m, r in sorted(zip(months, arr), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {m}: {r*100:+.2f}%")


def cmd_hold(trades, args):
    holds = np.array([t[5] for t in trades])
    print(f"持仓时间分布（{len(holds)} 笔）：")
    print(f"  均值 {holds.mean():.1f} 天 | 中位 {np.median(holds):.0f} 天 | 最短 {holds.min():.0f} | 最长 {holds.max():.0f}")
    print(f"  分位 P10 {np.quantile(holds,0.1):.0f} | P25 {np.quantile(holds,0.25):.0f} | P75 {np.quantile(holds,0.75):.0f} | P90 {np.quantile(holds,0.9):.0f}")
    print(f"  短持仓(≤10天) {(holds<=10).mean():.0%} | 长持仓(>30天) {(holds>30).mean():.0%} | 超时(60天) {(holds>=60).mean():.0%}")


def cmd_yearly(trades, args):
    yearly = {}
    for et, *_ , net_r, _ in trades:
        yearly.setdefault(et[:4], []).append(net_r)
    print(f"按年分解（{len(yearly)} 年，各年池化 g）：")
    pos = 0
    for y in sorted(yearly):
        Rs = yearly[y]
        valid = [r for r in Rs if 1 + F * r > 0]
        g = sum(math.log(1 + F * r) for r in valid) / len(valid) if valid else 0
        if g > 0:
            pos += 1
        print(f"  {y}: {len(Rs):>3} 笔  g={g*100:>+7.3f}%")
    print(f"→ 正年数 {pos}/{len(yearly)}")


def cmd_symbols(trades, args):
    by_sym = {}
    for _, sym, _, _, net_r, _ in trades:
        by_sym.setdefault(sym, []).append(net_r)
    print(f"各标的表现（{len(by_sym)} 个，按 g 排名）：")
    rows = []
    for sym, Rs in by_sym.items():
        valid = [r for r in Rs if 1 + F * r > 0]
        g = sum(math.log(1 + F * r) for r in valid) / len(valid) if valid else 0
        rows.append((sym, g, len(Rs), sum(Rs) / len(Rs)))
    for sym, g, n, ev in sorted(rows, key=lambda x: x[1], reverse=True):
        print(f"  {sym:<9} g={g*100:>+7.3f}%  n={n:>3}  EV净={ev:>+6.3f}")


def cmd_worst_month(trades, args):
    rets, months = monthly_returns(trades)
    worst = sorted(zip(months, rets), key=lambda x: x[1])[:3]
    for m, ret in worst:
        print(f"\n■ {m}（月收益 {ret*100:+.2f}%）：")
        mt = [t for t in trades if t[0][:7] == m]
        Rs = [t[3] for t in mt]
        print(f"  该月 {len(mt)} 笔：平均 R {np.mean(Rs):+.2f} | 最大亏损 {min(Rs):+.2f}R | 亏损 {(np.array(Rs)<0).mean():.0%} | 跳空穿透(R<-1) {sum(1 for r in Rs if r<-1)} 笔")
        for t in sorted(mt, key=lambda x: x[3])[:5]:
            print(f"    {t[1]:<9} {t[0][:10]}  R={t[3]:+.2f}  持仓{t[5]}天")


def cmd_gap(trades, args):
    gross = np.array([t[3] for t in trades])
    print(f"跳空穿透统计（R<-1，{len(gross)} 笔）：")
    print(f"  跳空穿透 {(gross<-1).sum()} 笔 ({(gross<-1).mean():.1%}) | 最差 {gross.min():+.2f}R | P1 {np.quantile(gross,0.01):+.2f}R")
    print(f"  完整止损（R≈-1）占比 {((gross>-1.02)&(gross<-0.98)).mean():.1%}")
    print(f"  最差 5 笔：")
    for t in sorted(trades, key=lambda x: x[3])[:5]:
        print(f"    {t[1]:<9} {t[0][:10]}  R={t[3]:+.2f}")


def cmd_signal_day(trades, args):
    """信号次日（t+1 进场日）的价格行为：开盘 vs VWAP/收盘、回踩触及率。"""
    US = sorted({t[1] for t in trades})
    days = [(t[1], t[0][:10]) for t in trades]
    n, hit_05, hit_10 = 0, 0, 0
    ovc, ova, lvo = [], [], []
    for sym, day in days:
        try:
            dm = _load(common.load_cache(sym, "15m"), "15m")
            day_min = dm[dm.index.date == pd.Timestamp(day).date()]
            if len(day_min) < 5:
                continue
        except Exception:
            continue
        n += 1
        o = float(day_min["open"].iloc[0])
        low = float(day_min["low"].min())
        high = float(day_min["high"].max())
        close = float(day_min["close"].iloc[-1])
        avg = float((day_min["close"] * day_min["volume"]).sum() / day_min["volume"].sum())
        if low >= o * (1 - 0.005):
            hit_05 += 1
        if low >= o * (1 - 0.010):
            hit_10 += 1
        ovc.append(close / o - 1)
        ova.append(avg / o - 1)
        lvo.append(low / o - 1)
    ovc, ova, lvo = map(np.array, (ovc, ova, lvo))
    print(f"信号次日价格行为（{n} 个信号日，需 15m 数据）:")
    print(f"  开盘 vs 收盘：均值 {ovc.mean()*100:+.2f}%（正=开盘买贵了）")
    print(f"  开盘 vs VWAP：均值 {ova.mean()*100:+.2f}%")
    print(f"  日内最低 vs 开盘：均值 {lvo.mean()*100:+.2f}%（盘中回踩深度）")
    print(f"  回踩 ≥0.5% 触及率：{(1-hit_05/n):.0%} | 回踩 ≥1.0% 触及率：{(1-hit_10/n):.0%}")
    print(f"  结论：开盘 vs VWAP 差 {ova.mean()*100:+.2f}% → 开盘买{'接近最优' if abs(ova.mean())<0.1 else '需评估延迟买'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="analyze.py", description="交易数据各维度分析")
    parser.add_argument("cmd", choices=["monthly", "hold", "yearly", "symbols", "worst-month", "gap", "signal-day"],
                        help="分析维度")
    parser.add_argument("--scheme", default="1", choices=["1", "2", "3", "4"], help="方案编号（默认 1）")
    parser.add_argument("--symbols", default=None, help="自定义标的（逗号分隔）")
    args = parser.parse_args()

    name, symbols, params, _ = common.SCHEMES[args.scheme]
    if args.symbols:
        symbols = [common.norm_symbol(s) for s in args.symbols.split(",")]
    trades = collect(symbols, params)
    print(f"分析：{name}（{len(trades)} 笔交易）\n")

    handlers = {
        "monthly": cmd_monthly, "hold": cmd_hold, "yearly": cmd_yearly,
        "symbols": cmd_symbols, "worst-month": cmd_worst_month,
        "gap": cmd_gap, "signal-day": cmd_signal_day,
    }
    handlers[args.cmd](trades, args)


if __name__ == "__main__":
    main()
