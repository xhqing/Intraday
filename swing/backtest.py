#!/usr/bin/env python3
"""回测 CLI：对方案（或自定义标的/模式）跑历史回测，输出全部关键维度指标。

用法（在 swing/ 目录下）：
  # 回测方案1（推荐交付方案：只做多 37 美股，全历史 ~20 年）
  python3 backtest.py --scheme 1

  # 对比方案 1-4
  python3 backtest.py --all

  # 自定义标的 + 模式（如只测 NVDA 做多）
  python3 backtest.py --symbols NVDA --mode long

输出：池化 g / EV / 笔数 / 正标的 / 按年正 / 月收益率统计 / 持仓分布 / 最差 R / 跳空穿透。
指标含义见 STRATEGY.md §7。
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import common
from backtest import run_backtest

F = 0.02  # g 的报告基准（每笔风险占权益 2%）


def collect_trades(symbols, params):
    """跑回测收集所有交易，返回 (trades, 正g标的数)。"""
    trades = []
    pos_sym = 0
    for sym in symbols:
        try:
            df = common.load_day(sym)
        except Exception:
            continue
        if len(df) < 200:
            continue
        r = run_backtest(trend_breakout(), sym, "DAY", params=params, df=df.copy())
        for t in r.trades:
            cost = common.MARKET_COST_BPS[common.market_of(sym)]
            net = t.R - (t.entry_price + t.exit_price) * cost * 1e-4 / t.risk
            trades.append((t.entry_time, t.symbol, t.R, net, t.hold_bars))
        if r.log_growth() > 0:
            pos_sym += 1
    return trades, pos_sym


def trend_breakout():
    """返回策略模块（延迟 import 避免循环）。"""
    from strategies import trend_breakout as tb
    return tb


def analyze(trades, symbols):
    """从交易列表计算全部维度指标。"""
    n = len(trades)
    net = [t[3] for t in trades]
    gross = [t[2] for t in trades]
    holds = [t[4] for t in trades]
    valid = [r for r in net if 1 + F * r > 0]
    g = sum(math.log(1 + F * r) for r in valid) / len(valid) if valid else 0
    ev = sum(net) / n if n else 0
    # 按年
    yearly = {}
    for et, _, _, net_r, _ in trades:
        yearly.setdefault(et[:4], []).append(net_r)
    pos_years = sum(1 for y in yearly if sum(yearly[y]) > 0)
    # 月收益率（f=2% 复利按月）
    srt = sorted(trades, key=lambda x: x[0])
    W, me = 1.0, {}
    for et, _, _, net_r, _ in srt:
        W *= 1 + F * net_r
        me[et[:7]] = W
    months = sorted(me)
    rets = np.array([me[m] / (me[months[i - 1]] if i > 0 else 1.0) - 1 for i, m in enumerate(months)])
    return {
        "n": n, "g": g, "ev": ev, "pos_sym": 0, "n_sym": len(symbols),
        "pos_years": pos_years, "n_years": len(yearly),
        "m_mean": rets.mean(), "m_med": np.median(rets), "m_std": rets.std(),
        "m_min": rets.min(), "m_max": rets.max(),
        "m_win": float((rets >= 0).sum()) / len(rets),
        "sharpe": rets.mean() / rets.std() * (12 ** 0.5) if rets.std() > 0 else 0,
        "hold_mean": float(np.mean(holds)), "hold_med": float(np.median(holds)),
        "worst": min(gross), "gap": sum(1 for r in gross if r < -1) / n if n else 0,
        "annual": n / (len(yearly) if yearly else 1),
    }


def print_report(name, d):
    print(f"\n■ {name}")
    print(f"  交易 {d['n']} 笔 | g@2% {d['g']*100:+.3f}% | EV净 {d['ev']:+.3f}R | 正标的 {d['pos_sym']}/{d['n_sym']} | "
          f"按年正 {d['pos_years']}/{d['n_years']}")
    print(f"  月均 {d['m_mean']*100:+.2f}% | 月std {d['m_std']*100:.2f}% | 月胜率 {d['m_win']:.0%} | "
          f"最差月 {d['m_min']*100:+.2f}% | 最好月 {d['m_max']*100:+.2f}%")
    print(f"  夏普 {d['sharpe']:.2f} | 持仓均值/中位 {d['hold_mean']:.1f}/{d['hold_med']:.0f} 天 | "
          f"最差R {d['worst']:+.2f} | 跳空穿透 {d['gap']:.1%} | 年化笔数 {d['annual']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="backtest.py", description="日 K 趋势跟随回测（方案/自定义）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scheme", default="1", choices=["1", "2", "3", "4"], help="方案编号（默认 1）")
    group.add_argument("--all", action="store_true", help="对比全部 4 个方案")
    parser.add_argument("--symbols", default=None, help="自定义标的（逗号分隔，自动补前缀）")
    parser.add_argument("--mode", choices=["long", "both"], default=None, help="自定义模式（long/both）")
    parser.add_argument("--params", default=None, help="额外参数 JSON，如 '{\"trail_atr\":4}'")
    args = parser.parse_args()

    import json as _json

    if args.all:
        for num in ["1", "2", "3", "4"]:
            name, symbols, params, _ = common.SCHEMES[num]
            trades, pos = collect_trades(symbols, params)
            d = analyze(trades, symbols)
            d["pos_sym"] = pos
            print_report(name, d)
        return

    name, symbols, params, desc = common.SCHEMES[args.scheme]
    if args.symbols:
        symbols = [common.norm_symbol(s) for s in args.symbols.split(",")]
        name = f"自定义 {len(symbols)} 标的"
    if args.mode:
        params = {**params, "mode": args.mode}
        name += f"（{args.mode}）"
    if args.params:
        params = {**params, **_json.loads(args.params)}
    print(f"回测：{name}（{desc}）")
    trades, pos = collect_trades(symbols, params)
    d = analyze(trades, symbols)
    d["pos_sym"] = pos
    print_report(name, d)


if __name__ == "__main__":
    main()
