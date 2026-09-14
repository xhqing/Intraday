#!/usr/bin/env python3
"""A 股跨境 ETF 开盘反向修正：执行严谨化回测（A 股线实验 2，2026-09-13）。

初筛（overnight_map.py）发现的信号：隔夜美股大涨大跌日，A 股跨境 ETF 开盘跳空过度、
开盘后半小时反向修正（513050 反向 +12~+18bps、7 年稳定）。但初筛口径混入了不可执行的
一半——A 股 ETF 不能裸空，「美股大涨日做空高开」无法交易，**只有「美股大跌日买入低开
等反弹」（做多半）可执行**。本脚本分解多空两半并做执行口径严谨化。

严谨化清单：
  ① 多空分解：可执行（r_us < −thr 做多）vs 不可执行（r_us > +thr 做空）各自贡献
  ② 持有时长扫描：15 / 30 / 60 / 90 分钟 / 持有到收盘（跨午休按时间索引处理）
  ③ 执行价格敏感性：9:31 开盘瞬间 / 9:32 / 9:35 进场（开盘流动性差，晚进场更保守）
  ④ 成本敏感性：佣金双边 2bps + 单边滑点 0 / 2 / 5bps
  ⑤ 分年收益 + t 值、收益分布（P5 / P50 / P95、最差单日）
  ⑥ 非对称分解：大跌日做多在大跌幅分档（−1% / −1.5% / −2%）上的单调性

用法：python3 etf_rev.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

QC = Path(__file__).parent.parent / ".claude" / "skills" / "quant" / "data_cache"
HOLDS = [15, 30, 60, 90, "close"]


def load_us() -> pd.DataFrame:
    df = pd.read_parquet(QC / "US.QQQ" / "DAY.parquet")
    df["us_date"] = pd.to_datetime(df["time"]).dt.tz_convert("Asia/Shanghai").dt.date
    df["r_us"] = df.sort_values("us_date")["close"].pct_change()
    return df[["us_date", "r_us"]].dropna()


def build_days(sym: str, us: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(QC / sym / "1m.parquet").set_index("time").sort_index()
    rows = []
    for d, g in df.groupby(df.index.date):
        g = g.sort_index()
        if len(g) < 30:
            continue
        t = g.index
        rec = {"date": d}
        for hh, mm, tag in ((9, 31, "e31"), (9, 32, "e32"), (9, 35, "e35")):
            m = g[(t.hour == hh) & (t.minute == mm)]
            if len(m):
                rec[f"px_{tag}"] = m["open"].iloc[0]
        for h in HOLDS:
            if h == "close":
                rec["px_close"] = g["close"].iloc[-1]
            else:
                tgt = t[0] + pd.Timedelta(minutes=h)
                sub = g[g.index <= tgt]
                if len(sub):
                    rec[f"px_{h}"] = sub["close"].iloc[-1]
        if all(np.isfinite(rec.get(k, np.nan)) for k in ("px_e31", "px_e35", "px_30", "px_close")):
            rows.append(rec)
    m = pd.DataFrame(rows).sort_values("date")
    us_dates = pd.to_datetime(us["us_date"]).values.astype("datetime64[D]")
    d_dates = pd.to_datetime(pd.to_datetime(m["date"]).dt.date).values.astype("datetime64[D]")
    idx = np.searchsorted(us_dates, d_dates, side="left") - 1
    m = m[idx >= 0].copy()
    m["r_us"] = us["r_us"].values[idx[idx >= 0]]
    return m.dropna(subset=["r_us"])


def stats(net: pd.Series, label: str):
    if len(net) == 0:
        print(f"    {label}: n=0")
        return
    t = net.mean() / (net.std() / np.sqrt(len(net)))
    q = net.quantile([0.05, 0.5, 0.95]) * 1e4
    print(f"    {label}: {net.mean()*1e4:+6.1f} bps | n={len(net):>4} | t={t:+.2f} | "
          f"P5 {q[0.05]:+.0f} / P50 {q[0.5]:+.0f} / P95 {q[0.95]:+.0f} | 最差 {net.min()*1e4:+.0f}")


def run(sym: str, us: pd.DataFrame):
    m = build_days(sym, us)
    print(f"\n=== {sym}（{len(m)} 日）===")

    # ① 多空分解（持 30 分钟、9:31 进场、双边 2bps）
    for thr in (0.01, 0.015):
        lo, hi = m[m["r_us"] < -thr], m[m["r_us"] > thr]
        print(f"  |r_us|>{thr:.1%} 多空分解（持有 30min）：")
        stats(lo["px_30"] / lo["px_e31"] - 1 - 2e-4, "可执行：大跌日买入（r_us<−t）")
        stats(-(hi["px_30"] / hi["px_e31"] - 1) - 2e-4, "不可执行：大涨日做空（r_us>+t）")

    # ② 持有时长扫描（只保留可执行半，thr=1.5%）
    print("  持有时长扫描（大跌日买入，9:31 进场，双边 2bps）：")
    for h in HOLDS:
        sel = m[m["r_us"] < -0.015]
        if h == "close":
            net = sel["px_close"] / sel["px_e31"] - 1 - 2e-4
            stats(net, "持有到收盘        ")
        else:
            net = sel[f"px_{h}"] / sel["px_e31"] - 1 - 2e-4
            stats(net, f"持有 {h:>3} 分钟      ")

    # ③ 执行价格敏感性（持有 30 分钟）
    print("  执行价格敏感性（大跌日买入，持 30min，双边 2bps）：")
    for tag, name in (("e31", "9:31 开盘"), ("e32", "9:32 "), ("e35", "9:35 ")):
        sel = m[m["r_us"] < -0.015]
        net = sel["px_30"] / sel[f"px_{tag}"] - 1 - 2e-4
        stats(net, f"{name}进场")

    # ④ 成本敏感性（9:32 进场、持 30min）
    print("  成本敏感性（9:32 进场、持 30min，佣金 2bps + 单边滑点）：")
    sel = m[m["r_us"] < -0.015]
    for slip in (0, 2e-4, 5e-4):
        net = sel["px_30"] / sel["px_e32"] - 1 - 2e-4 - slip
        stats(net, f"滑点 {slip*1e4:.0f}bps")

    # ⑤ 分年（可执行口径：大跌日买入、9:32 进、持 30min、双边 2bps+滑点 2bps）
    sel = m[m["r_us"] < -0.015].copy()
    sel["net"] = sel["px_30"] / sel["px_e32"] - 1 - 4e-4
    sel["y"] = pd.to_datetime(sel["date"]).dt.year
    print("  分年（9:32 进、持 30min、双边 2bps+滑点 2bps）：")
    for y, g in sel.groupby("y"):
        t = g["net"].mean() / (g["net"].std() / np.sqrt(len(g)))
        print(f"    {y}: {g['net'].mean()*1e4:+6.1f} bps | n={len(g):>3} | t={t:+.2f}")

    # ⑥ 大跌幅分档单调性
    print("  大跌幅分档（9:32 进、持 30min、双边 2bps+滑点 2bps）：")
    bins = [(-0.03, -0.02), (-0.02, -0.015), (-0.015, -0.01), (-0.01, -0.005)]
    for a, b in bins:
        g = m[(m["r_us"] >= a) & (m["r_us"] < b)]
        if len(g) < 10:
            continue
        net = g["px_30"] / g["px_e32"] - 1 - 4e-4
        stats(net, f"r_us ∈ [{a:.1%}, {b:.1%})")


if __name__ == "__main__":
    us = load_us()
    for sym in ("SH.513050", "SZ.159920", "SH.513100"):
        run(sym, us)
