#!/usr/bin/env python3
"""限价单模拟实验：被动挂单能否吃到「存活窗口短于一分钟」的 edge（回测方法改进实验 4）。

背景：信号分钟 t 收盘进场 +24.0 bps，但 t+1 开盘市价进场 -4.8 bps——主动吃延迟拿不到 edge。
若 edge 真在信号后瞬时存在，正确的执行可能是被动挂单（等价格回到信号价成交，不付延迟）。

设计（与 verify_2025 完全同口径：H=6、同特征、同超参数、同窗口切分、零调参）：
  - 信号：t 分钟收盘出信号（q∈{最空,最多}），挂单价基于 close_t（t 收盘后已知，无未来函数）
  - 挂单：t+1 分钟内挂限价单，做多 limit = close_t - δ、做空 limit = close_t + δ，
    δ 扫描 {0, 1, 2, 5, 10} 美分（QQQ tick=0.01，2025 年价格约 480~530，0.01≈0.2bps）
  - 未触及则撤单（收益记 0，不交易）
  - 成交判定两口径（委托价口径 low/high 比成交价宽，fill rate 是上界；严格口径作修正）：
      touch：做多 low[t+1] ≤ limit / 做空 high[t+1] ≥ limit
      strict：做多 low[t+1] < limit - 0.005（需穿过多于半 tick）/ 做空对称
  - 出场：t+6 收盘市价（原持仓窗口不变），成本 = maker 进 0 + taker 出 3bps
    （原口径双边 6bps；被动进场省一半成本）

一次一仓：不重叠采样下信号间隔 = 持仓长度 6 分钟，信号天然不重叠，逐信号独立模拟即可。

指标：fill rate、条件净收益/笔（成交的）、每信号期望（未成交记 0）。
基准：r_close +24.0 bps（回测口径）/ r_next -4.8 bps（t+1 市价口径）。

用法（分年跑防 swap）：
  python3 limit_sim.py --year 2025
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, H, TRAIN_DAYS, TEST_DAYS, PARAMS

TMP = Path(__file__).parent.parent / "tmp"
DELTAS = [0.0, 0.01, 0.02, 0.05, 0.10]   # 挂单偏移（美元）
STRICT_BUF = 0.005                        # 严格口径缓冲（半 tick）
TAKER = 3e-4                              # 出场单边成本 3bps


def run_year(year: str) -> None:
    min_df = add_atr(load_minute_year(year))
    print(f"{year}: {len(min_df)} 分钟", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    close_all = min_df["close"].values
    low_all = min_df["low"].values
    high_all = min_df["high"].values
    dates = min_df.index
    n = len(min_df)

    # 每个信号的模拟结果：{delta: {touch:[r...], strict:[r...]}}，NaN=未成交（记 0 于汇总时）
    rows = []
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        te_e, te_x = e_idx[te], x_idx[te]
        for i in range(len(te_e)):
            if q[i] not in (0, 4):
                continue
            t0, t1, tx = int(te_e[i]), int(te_x[i]) - H + 1, int(te_x[i])
            pos = 1 if q[i] == 4 else -1
            c0 = close_all[t0]
            # 挂单分钟与出场分钟越界 / 跨日（挂单跨日不现实，撤单）
            if t1 >= n or dates[t1].date() != dates[t0].date():
                continue
            exit_p = close_all[tx]
            row = {"date": str(dates[t0].date()), "pos": pos, "r_close_ref": None}
            for d in DELTAS:
                limit = c0 - d if pos == 1 else c0 + d
                touch = (low_all[t1] <= limit) if pos == 1 else (high_all[t1] >= limit)
                strict = (low_all[t1] < limit - STRICT_BUF) if pos == 1 else (high_all[t1] > limit + STRICT_BUF)
                for tag, hit in (("touch", touch), ("strict", strict)):
                    key = f"d{int(d*100):03d}_{tag}"
                    if hit and limit > 0:
                        row[key] = pos * (exit_p - limit) / limit - TAKER * exit_p / limit
                    else:
                        row[key] = np.nan
            rows.append(row)
        if wid % 20 == 0:
            print(f"  窗口 {wid}/{len(slices)-1}，信号 {len(rows)} 个", flush=True)

    df = pd.DataFrame(rows)
    out_csv = TMP / f"limit_sim_{year}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n=== {year} 限价单模拟（信号 {len(df)} 个，成本 maker 0 + taker 3bps）===")
    print(f"{'δ(美分)':>8} | {'touch fill':>10} | {'touch 条件bps':>12} | {'touch 期望bps':>12} | "
          f"{'strict fill':>11} | {'strict 条件bps':>13} | {'strict 期望bps':>13}")
    summary = {}
    for d in DELTAS:
        s = {}
        parts = [f"{int(d*100):>8.0f}"]
        for tag in ("touch", "strict"):
            r = df[f"d{int(d*100):03d}_{tag}"]
            fill = r.notna()
            fr = fill.mean()
            cond = r.mean() * 1e4 if fill.any() else float("nan")
            exp = (r.fillna(0)).mean() * 1e4
            s[tag] = {"fill_rate": float(fr), "cond_bps": float(cond), "exp_bps": float(exp),
                      "n_fill": int(fill.sum())}
            parts.append(f"{fr*100:>9.1f}% {cond:>+12.1f} {exp:>+12.1f}")
        summary[f"{d:.2f}"] = s
        print(" | ".join(parts))
    print("\n基准：信号分钟收盘市价进 +24.0 bps（回测口径）| t+1 开盘市价进 -4.8 bps（主动吃延迟）")
    out_json = TMP / f"limit_sim_{year}.json"
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"明细 {out_csv}｜汇总 {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], required=True)
    args = parser.parse_args()
    run_year(args.year)
