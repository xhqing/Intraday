#!/usr/bin/env python3
"""检测信号 CLI：每天收盘后判断哪些标的触发日 K 趋势跟随做多信号。

用法（在 swing/ 目录下）：
  # 检测全部 37 只标的（默认，每天美股收盘后跑）
  python3 check_signal.py

  # 指定标的（自动补 US. 前缀）
  python3 check_signal.py --symbols SPY,NVDA,AAPL

  # 详细模式（无信号时显示距离触发差多少）
  python3 signal.py --symbols NVDA --verbose

  # 回看历史某天
  python3 signal.py --symbol NVDA --date 2026-08-04

  # 用其他方案（如方案2 多空都做）
  python3 signal.py --scheme 2

输出：✅ 触发信号 → 显示信号详情 + 该类别历史置信度；❌ 无信号 → 显示当前指标距离触发差多少。
策略逻辑详见 STRATEGY.md。
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import common
from backtest import add_indicators, _load
from strategies import trend_breakout

CONF_FILE = common.QUANT_DIR / "confidence_table.json"


def load_confidence() -> dict:
    """读回测产出的「类别 → 可信度」查表（quant/confidence_table.json，回测时生成）。"""
    if CONF_FILE.exists():
        return json.loads(CONF_FILE.read_text(encoding="utf-8"))
    return {}


def check_symbol(symbol: str, params: dict, date_str: str | None, conf: dict, verbose: bool) -> bool:
    """检测单个标的是否触发信号。返回是否触发。"""
    df = common.load_day(symbol)
    df = add_indicators(_load(df, "DAY"), params)

    if date_str:
        target = pd.Timestamp(date_str)
        if df.index.tz is not None and target.tzinfo is None:
            target = target.tz_localize(df.index.tz)
        idx = df.index.get_indexer([target], method="ffill")[0]
        bars = df.iloc[: idx + 1]
    else:
        bars = df

    last = bars.iloc[-1]
    signal = trend_breakout.evaluate(bars, params)

    if signal:
        cat = signal["category"]
        row = conf.get(cat, {})
        signal["confidence"] = {
            "win_rate": row.get("win_rate"), "avg_R": row.get("avg_R"),
            "samples": row.get("samples"), "low_sample": row.get("low_sample"),
        }
        print(f"✅ {common.norm_symbol(symbol)} {last.name.date()} 触发信号（{cat}）：")
        print(f"   方向 {'做多' if signal['direction'] > 0 else '做空'} | "
              f"历史胜率 {row.get('win_rate', '-')} | avg_R {row.get('avg_R', '-')} | 样本 {row.get('samples', '-')}")
        print(f"   → 次日开盘执行；止损 = 开盘价 ∓ 2×ATR，3×ATR trailing（见 STRATEGY.md §4-5）")
        return True

    print(f"❌ {common.norm_symbol(symbol)} {last.name.date()} 无信号")
    if verbose:
        recent_high = float(bars["close"].iloc[-21:-1].max())
        close = float(last["close"])
        vr = float(last["vol_ratio"] or 0.0)
        print(f"   收盘 {close:.2f} | 需突破 {recent_high:.2f}（价差 {close - recent_high:+.2f}）| "
              f"量比 {vr:.2f}（需 ≥1.5）")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="signal.py", description="日 K 趋势跟随信号检测（只读，无副作用）")
    parser.add_argument("--symbols", default=None, help="标的列表，逗号分隔（默认方案标的池）")
    parser.add_argument("--scheme", default="1", choices=["1", "2", "3", "4"], help="方案编号（默认 1 只做多）")
    parser.add_argument("--date", default=None, help="回看日期 YYYY-MM-DD（默认最新交易日）")
    parser.add_argument("--verbose", action="store_true", help="无信号时显示距离触发差多少")
    args = parser.parse_args()

    name, symbols, params, _ = common.SCHEMES[args.scheme]
    if args.symbols:
        symbols = [common.norm_symbol(s) for s in args.symbols.split(",")]

    conf = load_confidence()
    hit = 0
    for sym in symbols:
        try:
            if check_symbol(sym, params, args.date, conf, args.verbose):
                hit += 1
        except Exception as e:
            print(f"⚠️ {sym}: {e}")
    print(f"\n{hit}/{len(symbols)} 个标的触发信号（方案{args.scheme} {name}）。有信号 → 次日开盘执行（仓位见 STRATEGY.md §8）。")


if __name__ == "__main__":
    main()
