#!/usr/bin/env python3
"""实盘经济学测算：mbo 口径 edge 值不值 Databento Plus 订阅（~$1,780/月）。

与 verify_2025.py 完全同口径（同特征、同超参数、同窗口切分、零调参），但记录每笔交易明细：
  - 日期、方向、净 R（按 risk=entry 口径）
  - 两种进场成交假设：信号分钟收盘价（回测口径）vs 下一分钟开盘价（更接近实盘——
    盘中实时出信号后下单，实际成交在下一根 K 线开盘附近）
明细落盘 tmp/trades_<year>.csv，供后续财务测算（月度盈亏 / 保本仓位 / 凯利 / 回撤）。

用法（分年跑防 swap，n_jobs=2 不占满机器）：
  python3 economics.py --year 2025
  python3 economics.py --year 2026
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, H, TRAIN_DAYS, TEST_DAYS, PARAMS

TMP = Path(__file__).parent.parent / "tmp"


def run_year(year: str) -> None:
    min_df = add_atr(load_minute_year(year))
    print(f"{year}: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    close_all = min_df["close"].values
    open_all = min_df["open"].values
    dates = min_df.index

    records = []
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        # 一次一仓 + 持仓 H 分钟 + 无止损（risk=entry，与验证口径一致）
        pos, pos_end_min, t0 = 0, -1, 0
        te_e, te_x = e_idx[te], x_idx[te]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                pos_end_min = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end_min:
                exit_p = close_all[pos_end_min]
                # 进场假设 1：信号分钟收盘（原回测口径）
                e1 = close_all[t0]
                r1 = pos * (exit_p - e1) / e1 - 3e-4 * (e1 + exit_p) / e1
                # 进场假设 2：下一分钟开盘（实盘更接近——出信号后下单，t+1 分钟成交）
                e2 = open_all[t0 + 1] if t0 + 1 < len(open_all) else float("nan")
                if not (np.isfinite(e2) and e2 > 0):
                    records.append({"date": dates[t0].date(), "pos": pos,
                                    "r_close": r1, "r_next": float("nan")})
                    pos = 0
                    continue
                r2 = pos * (exit_p - e2) / e2 - 3e-4 * (e2 + exit_p) / e2
                records.append({"date": dates[t0].date(), "pos": pos, "r_close": r1, "r_next": r2})
                pos = 0
        if wid % 20 == 0:
            print(f"  窗口 {wid}/{len(slices)-1}，累计 {len(records)} 笔", flush=True)

    TMP.mkdir(exist_ok=True)
    out = TMP / f"trades_{year}.csv"
    pd.DataFrame(records).to_csv(out, index=False)

    df = pd.DataFrame(records)
    for col, name in [("r_close", "信号分钟收盘进场（回测口径）"), ("r_next", "下一分钟开盘进场（实盘近似）")]:
        r = df[col]
        print(f"\n{year} {name}: 笔数 {len(r)} | 均值 {r.mean()*1e4:+.1f} bps | 中位 {r.median()*1e4:+.1f} | "
              f"std {r.std()*1e4:.1f} | 胜率 {(r>0).mean()*100:.1f}%")
    print(f"明细已存 {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], required=True)
    args = parser.parse_args()
    run_year(args.year)
