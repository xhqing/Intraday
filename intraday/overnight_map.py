#!/usr/bin/env python3
"""隔夜映射检验：隔夜 QQQ 行情 → A 股跨境 ETF 开盘定价效率（A 股结构性溢价线实验 1，2026-09-13）。

背景：A 股跨境 ETF（T+0 品种）方向的第一候选机制不是日内方向预测，而是「隔夜信息的
映射效率」——A 股 9:30 开盘时，隔夜美股（约 5.5 小时前收盘）信息是否被充分定价：
反应不足 → 开盘后延续（可交易）；反应过度 → 开盘后回归（可反向）。这是制度性摩擦
（QDII 额度、参与者结构）而非预测力，绕开订单流研究的三层死因。

数据：富途 1m（A 股 ETF 6.5 年 + QQQ 同期）；日线对齐——A 股日 D 的「隔夜美股」=
收盘时刻早于 D 09:30（北京）的最近一个美东交易日（自然为前一交易日，节假错位取最近）。

变量（A 股日 D）：
  r_us  = 隔夜 QQQ 日收益（昨收→收）
  gap   = ETF D 开盘 / ETF D-1 收盘 − 1（跳空幅度）
  r_post30 = 9:31 open → 10:00 close（开盘后半小时）
  r_day = D-1 收盘 → D 收盘（全日）
检验：
  ① 映射效率：gap = a + b·r_us 的 b（≈1 充分定价 / >1 过度 / <1 不足）
  ② 开盘后动量 / 回归：corr(r_us, r_post30)、corr(gap, r_post30)
  ③ 扣成本交易：r_us 强（|r_us|>1%）时开盘追 / 反向，持有 30 分钟；
     A 股 ETF 佣金约 0.5~1bps/边，双边按 2bps 计。
  ④ 分年稳定性。

用法：python3 overnight_map.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

QC = Path(__file__).parent.parent / ".claude" / "skills" / "quant" / "data_cache"
COST = 2e-4   # A 股 ETF 双边佣金（0.5~1bps/边，取 2bps 保守）


def load_us_daily() -> pd.DataFrame:
    df = pd.read_parquet(QC / "US.QQQ" / "DAY.parquet")
    df["us_date"] = pd.to_datetime(df["time"]).dt.tz_convert("Asia/Shanghai").dt.date
    # 美股收盘的北京时刻（16:00 ET = 北京 04:00/05:00）恒早于同编号日的 A 股开盘
    df = df.sort_values("us_date")
    df["r_us"] = df["close"].pct_change()
    return df[["us_date", "close", "r_us"]].dropna()


def day_metrics(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(QC / sym / "1m.parquet")
    df = df.set_index("time").sort_index()
    rows = []
    for d, g in df.groupby(df.index.date):
        g = g.sort_index()
        if len(g) < 30:
            continue
        t = g.index
        m931 = g[(t.hour == 9) & (t.minute == 31)]
        m1000 = g[(t.hour == 10) & (t.minute == 0)]
        if len(m931) == 0 or len(m1000) == 0:
            continue
        o931, c1000, cday = m931["open"].iloc[0], m1000["close"].iloc[0], g["close"].iloc[-1]
        if any(v <= 0 for v in (o931, c1000, cday)):
            continue
        rows.append({"date": d, "o931": o931, "c1000": c1000, "cday": cday})
    m = pd.DataFrame(rows).sort_values("date")
    m["gap"] = m["o931"] / m["cday"].shift(1) - 1
    m["r_post30"] = m["c1000"] / m["o931"] - 1
    m["r_day"] = m["cday"] / m["cday"].shift(1) - 1
    return m.dropna()


def run(sym: str, us: pd.DataFrame):
    m = day_metrics(sym)
    # 对齐：A 股日 D 映射「收盘早于 D 开盘的最近美股日」——用 searchsorted
    us_dates = pd.to_datetime(us["us_date"]).values.astype("datetime64[D]")
    d_dates = pd.to_datetime(pd.to_datetime(m["date"]).dt.date).values.astype("datetime64[D]")
    idx = np.searchsorted(us_dates, d_dates, side="left") - 1
    ok = idx >= 0
    m = m[ok].copy()
    m["r_us"] = us["r_us"].values[idx[ok]]
    m = m.dropna(subset=["r_us"])
    print(f"\n=== {sym} 隔夜映射（{len(m)} 日，{m.date.min()} ~ {m.date.max()}）===")

    b, a = np.polyfit(m["r_us"], m["gap"], 1)
    r2 = np.corrcoef(m["r_us"], m["gap"])[0, 1] ** 2
    print(f"  映射效率 gap = {a*1e4:+.1f}bps + {b:.3f}·r_us（R²={r2:.3f}）"
          f"—— {'充分' if 0.8 < b < 1.2 else ('过度' if b >= 1.2 else '不足')}")

    for x, y in (("r_us", "r_post30"), ("gap", "r_post30")):
        c = m[x].corr(m[y])
        print(f"  corr({x}, r_post30): {c:+.4f}")

    # 开盘后动量交易：|r_us| > 1% 时按 r_us 方向 / 反向持有 30 分钟
    for thr, name in ((0.01, "|r_us|>1%"), (0.015, "|r_us|>1.5%")):
        sel = m[m["r_us"].abs() > thr]
        if len(sel) < 30:
            continue
        sig = np.sign(sel["r_us"])
        mom = (sig * sel["r_post30"] - COST).mean() * 1e4
        rev = (-sig * sel["r_post30"] - COST).mean() * 1e4
        print(f"  {name}（n={len(sel)}）: 开盘追 {mom:+.1f} bps | 开盘反向 {rev:+.1f} bps（扣双边 2bps）")
        # 分年（追方向）
        sel2 = sel.assign(y=pd.to_datetime(sel["date"]).dt.year,
                          net=sig * sel["r_post30"] - COST)
        by = sel2.groupby("y")["net"].agg(["mean", "count"])
        print("    追方向分年 bps: " + " ".join(f"{int(y)}:{v*1e4:+.1f}(n={int(n)})" for y, v, n in zip(by.index, by['mean'], by['count'])))


if __name__ == "__main__":
    us = load_us_daily()
    for sym in ("SH.513050", "SH.513100", "SZ.159920"):
        run(sym, us)
