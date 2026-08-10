#!/usr/bin/env python3
"""共享模块：quant 数据层/回测引擎的复用 + 标的池 + 策略参数。三个 CLI（signal/backtest/analyze）共用。

解耦原则：本模块是 CLI 与 quant skill（数据/回测引擎）之间的唯一接缝——
  - CLI 层（signal.py/backtest.py/analyze.py）只管用户命令与输出
  - 数据与回测复用 quant 的 data.py/backtest.py（同一 data_cache、同一引擎，保证口径一致）
  - 三个 CLI 互不依赖，只依赖本模块
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- quant skill 目录（数据层 + 回测引擎的权威实现）----
QUANT_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "quant"
if str(QUANT_DIR) not in sys.path:
    sys.path.insert(0, str(QUANT_DIR))

# ---- 复用 quant 的数据层与回测引擎 ----
from data import load_cache, DataError, market_of        # noqa: E402
from backtest import (                                   # noqa: E402
    _load, add_indicators, DEFAULT_PARAMS, run_backtest,
    MARKET_COST_BPS, BacktestResult, confidence_table,
)

# ---- 标的池（方案1：只做多 + 37 只纯美股）----
STOCK_POOL_37 = [
    "SPY", "QQQ", "DIA", "IWM",                                    # ETF
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX",  # 科技
    "AMD", "INTC", "ORCL", "IBM", "CSCO",
    "JPM", "GS", "MS", "V", "MA", "BAC",                            # 金融
    "WMT", "PG", "KO", "MCD", "DIS", "NKE",                         # 消费
    "UNH", "JNJ", "PFE", "LLY",                                     # 医疗
    "XOM", "CVX",                                                    # 能源
    "CAT", "BA",                                                     # 工业
]
LOW_CORR = ["GLD", "TLT", "IEF", "DBC", "VNQ", "EEM"]   # 低相关 ETF（方案3/4）
HK_POOL = ["02800", "00005", "00001", "00388", "00688", "00700", "00939",
           "00941", "01109", "01299", "02318", "03690", "09988"]

# ---- 策略参数（方案1 推荐配置：只做多 + trailing 3）----
PARAMS_LONG = {**DEFAULT_PARAMS, "mom_period": 20, "vol_mult": 1.5, "stop_atr": 2.0,
               "take_atr": 1000.0, "trail_atr": 3.0, "max_hold_bars": 60, "mode": "long"}
PARAMS_BOTH = {**PARAMS_LONG, "mode": "both"}

# 方案定义：名称 → (标的列表, 参数, 描述)
SCHEMES = {
    "1": ("方案1 只做多美股37", [f"US.{s}" for s in STOCK_POOL_37], PARAMS_LONG,
          "只做多，37 只纯美股（推荐交付方案）"),
    "2": ("方案2 多空都做美股37", [f"US.{s}" for s in STOCK_POOL_37], PARAMS_BOTH,
          "多空都做，37 只纯美股（月胜率 65% 但波动翻倍）"),
    "3": ("方案3 只做多美股43含低相关", [f"US.{s}" for s in STOCK_POOL_37 + LOW_CORR], PARAMS_LONG,
          "只做多，37 美股 + 6 低相关 ETF（削危机月尾，g 略降）"),
    "4": ("方案4 多空都做美股43含低相关", [f"US.{s}" for s in STOCK_POOL_37 + LOW_CORR], PARAMS_BOTH,
          "多空都做，37 美股 + 6 低相关 ETF（夏普最高但尾部最差）"),
}


def norm_symbol(sym: str) -> str:
    """标准化符号：自动补 US./HK. 前缀。'NVDA' → 'US.NVDA'；'02800' → 'HK.02800'。"""
    s = sym.strip().upper()
    if "." in s:
        return s
    # 纯数字 = 港股，否则美股
    return f"HK.{s}" if s.isdigit() else f"US.{s}"


def load_day(symbol: str):
    """加载标的日 K（原始 df，有 time 列），供 run_backtest 使用。"""
    return load_cache(norm_symbol(symbol), "DAY")
