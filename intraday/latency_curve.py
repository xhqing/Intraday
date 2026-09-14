#!/usr/bin/env python3
"""延迟-期望曲线：主动进场的延迟衰减形态（回测方法改进实验 6）。

背景：主动市价口径唯一测过的点是「t+1 分钟开盘」−4.8bps（economics.py，1 分钟延迟），
但 0~60 秒之间的形态未知——衰减曲线（H=1 AUC 最强、每分钟 edge 密度 25.7bps）暗示
edge 大部分在信号后第一分钟，延迟多短才够是主动端生死问题（队列口径被动端已否定）。

设计（与 queue_sim 同批信号，直接复用其信号表）：
  - 进场：信号分钟收盘时刻（t1_ns）+ 延迟 d 秒后主动市价进场，
    成交价 = 该时刻之后的第一笔 F 成交价（displayed fill，市价单语义——
    我们的市价单立即吃对手档，成交价 ≈ 下一笔市场成交价；QQQ 盘中成交每秒数笔）
  - d 网格：{0, 0.5, 1, 2, 5, 10, 20, 40, 60} 秒
  - 出场：t+6 分钟收盘（委托价口径 close，与全管道一致）市价
  - 成本：双边 taker 6bps（3e-4 × (entry+exit)/entry，risk=entry，与 verify_2025 一致）
  - 延迟压缩持仓时长（60s 延迟时持仓剩 5 分钟）——这是延迟代价的一部分

参考线（同批 5090 信号，逐信号口径）：
  - 上限：信号分钟收盘价进场（close_t，fill 100% 幻觉）≈ +24bps
  - 对照：economics.py t+1 开盘（一次一仓 3596 笔，委托价 open 口径）= −4.8bps

用法（分年跑）：
  python3 latency_curve.py --year 2025
  python3 latency_curve.py --year 2026
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TMP = Path(__file__).parent.parent / "tmp"
MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
DELAYS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0]   # 秒
COST = 3e-4   # 单边 taker 3bps，双边 ×2（按 verify_2025 口径折算）


def load_day_fills(day_dir: Path):
    """当日全部 F 成交事件 → (ts_ns[], price[])（ts 升序）。"""
    parts = []
    for f in sorted(day_dir.glob("*.parquet")):
        if f.stat().st_size <= 1024:
            continue
        try:
            t = pq.read_table(f, columns=["action", "price", "ts_event"]).to_pandas()
            parts.append(t[t["action"] == "F"])
        except Exception as e:
            print(f"  ⚠️ 跳过 {f.name}: {str(e)[:60]}", flush=True)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    ts = df["ts_event"].astype("int64").values
    px = df["price"].values.astype(np.float64)
    order = np.argsort(ts, kind="mergesort")
    return ts[order], px[order]


def run_year(year: str) -> None:
    sigs = pd.read_csv(TMP / f"queue_signals_{year}.csv")
    day_dirs = sorted(d for d in MBODIR.iterdir() if d.is_dir() and d.name.startswith(year))
    rows = []
    for day_dir in day_dirs:
        sigs_day = sigs[sigs["date"] == day_dir.name[:10]]
        if len(sigs_day) == 0:
            continue
        fills = load_day_fills(day_dir)
        if fills is None:
            continue
        f_ts, f_px = fills
        for _, s in sigs_day.iterrows():
            t1 = int(s["t1_ns"])
            row = {"date": s["date"], "pos": s["pos"]}
            for d in DELAYS:
                tq = t1 + int(d * 1e9)
                i = int(np.searchsorted(f_ts, tq, side="left"))
                entry = f_px[min(i, len(f_px) - 1)]   # 越界兜底最后一笔
                ex = s["exit_p"]
                row[f"d{d:g}"] = s["pos"] * (ex - entry) / entry - COST * (entry + ex) / entry
            rows.append(row)
        print(f"  {day_dir.name[:10]}: 成交 {len(f_ts):,} 笔，信号累计 {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = TMP / f"latency_curve_{year}.csv"
    df.to_csv(out_csv, index=False)

    # 汇总
    summary = {}
    print(f"\n=== {year} 延迟-期望曲线（逐信号 {len(df)} 个，双边 6bps）===")
    print(f"{'延迟(秒)':>8} | {'期望bps':>8} | {'中位bps':>8} | {'胜率':>6}")
    for d in DELAYS:
        r = df[f"d{d:g}"]
        s = {"exp_bps": float(r.mean() * 1e4), "med_bps": float(r.median() * 1e4),
             "win_rate": float((r > 0).mean()), "n": int(len(r))}
        summary[f"{d:g}"] = s
        print(f"{d:>8g} | {s['exp_bps']:>+8.1f} | {s['med_bps']:>+8.1f} | {s['win_rate']*100:>5.1f}%")
    # 上限参考（同批信号 close_t 进场，fill 100%）
    up = sigs["pos"] * (sigs["exit_p"] - sigs["close_t"]) / sigs["close_t"] \
        - COST * (sigs["close_t"] + sigs["exit_p"]) / sigs["close_t"]
    summary["upper_close_t"] = {"exp_bps": float(up.mean() * 1e4), "n": int(len(up))}
    print(f"参考上限（close_t 进场，fill 100%）: {up.mean()*1e4:+.1f} bps")
    out_json = TMP / f"latency_curve_{year}.json"
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"明细 {out_csv}｜汇总 {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], required=True)
    args = parser.parse_args()
    run_year(args.year)
