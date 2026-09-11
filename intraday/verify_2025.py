#!/usr/bin/env python3
"""2025 年跨年验证：edge 是否在 2026 样本外成立（mbo 委托簿口径）。

背景：+30bps/笔 的 edge 只在 2026-06-27~08-05 的 26 个交易日上验证过（28 窗口）。
现补拉 2025-06-01~09-30（92 个交易日）做决定性检验：同参数（不调参、不选特征）在
2025 年上滚动训练测试，看 AUC 与净收益是否复现。

设计（防过拟合纪律的延续）：
- 完全复用 common_wf 的口径（同采样、同特征、同窗口切分、同超参数）
- 2025 与 2026 分别跑（不混训练），各自独立 walk-forward
- 判定：2025 年 AUC 与净收益方向、量级 vs 2026 年对照

用法：
  python3 verify_2025.py --year 2025   # 只跑 2025（验证组）
  python3 verify_2025.py --year 2026   # 只跑 2026（对照组）
  python3 verify_2025.py               # 两年连跑（8GB 内存机器建议分年跑防 swap）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import add_atr, build_nonoap, window_slices

MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
OUT = Path("/tmp") / "verify_2025_result.json"


def load_minute_year(year: str) -> pd.DataFrame:
    """只加载指定年份的 mbo 数据聚合到分钟（分年跑防 OOM/swap）。

    用 pyarrow.parquet.read_table 单文件读（磁盘 89% 满时 pd.read_parquet 的
    dataset API 会 Errno 60 读超时，单文件模式实测 24 段 1.1s）。
    """
    import os

    import pyarrow.parquet as pq

    from ml_orderflow_v2 import _agg_minute
    frames = []
    for day_dir in sorted(MBODIR.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith(year):
            continue
        files = sorted(day_dir.glob("*.parquet"))
        cols = ["action", "side", "price", "size", "ts_event"]   # _agg_minute 只用这几列
        parts = []
        for f in files:
            if os.path.getsize(f) <= 1024:
                continue
            try:
                parts.append(pq.read_table(f, columns=cols).to_pandas())
            except Exception as e:   # 磁盘满时偶发读坏/超时：跳过该段不中断整年
                print(f"  ⚠️ 跳过 {day_dir.name[:10]}/{f.name}: {str(e)[:60]}", flush=True)
        del files
        if not parts:
            continue
        try:
            frames.append(_agg_minute(pd.concat(parts)))
        except Exception as e:
            print(f"  ⚠️ 聚合失败 {day_dir.name[:10]}: {str(e)[:60]}", flush=True)
        del parts
    df = pd.concat(frames).sort_index()
    del frames
    # 坏点清洗：盘后无成交分钟被聚合器记为 OHLC=0（2025 实测 856 个/72 天，集中 UTC 21~23 时）。
    # 不剔会把「价格归零」喂进特征与 R 计算（做空假赚 100%，曾虚增净收益 43bps）
    n_bad = int((df["close"] <= 0).sum())
    df = df[df["close"] > 0]
    print(f"坏点清洗：剔除 close=0 分钟 {n_bad} 个", flush=True)
    return df

# 固定参数（与 wf_H6 完全一致，不调）
H = 6
TRAIN_DAYS, TEST_DAYS = 8, 2
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
COST_R_FACTOR = 3e-4          # 单边成本 bps × 双边（entry+exit）/risk，risk=entry 时即 3bps×2


def run_wf(min_df, label):
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    close_all = min_df["close"].values
    aucs, nets, n_trades = [], [], 0
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        aucs.append(auc)
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        # 一次一仓 + 持仓 H 分钟 + 无止损（risk=entry 口径，与定稿一致）
        trades = []
        pos, entry_p, pos_end_min, t0 = 0, 0.0, -1, 0
        te_e, te_x = e_idx[te], x_idx[te]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                entry_p = close_all[t0]
                pos_end_min = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end_min:
                exit_p = close_all[pos_end_min]
                gross_r = pos * (exit_p - entry_p) / entry_p
                cost_r = 3e-4 * (entry_p + exit_p) / entry_p
                trades.append(gross_r - cost_r)
                pos = 0
        n_trades += len(trades)
        if trades:
            nets.append(np.mean(trades))
    aucs_f = [a for a in aucs if np.isfinite(a)]
    print(f"\n=== {label}（{len(slices)} 窗口，H={H}，8 天训练 / 2 天测试，无止损 risk=entry）===")
    print(f"  AUC: 均值 {np.mean(aucs_f):.3f} | 正窗口(>0.52) {sum(1 for a in aucs_f if a > 0.52)}/{len(aucs_f)}")
    if nets:
        mean_net = np.mean(nets)
        pos_win = sum(1 for s in nets if s > 0)
        print(f"  净R/笔: {mean_net:+.4f}（{mean_net*1e4:+.1f} bps）| 正窗口 {pos_win}/{len(nets)} | 总交易 {n_trades}")
        verdict = "✅ edge 复现" if (mean_net > 0 and pos_win >= len(nets) * 0.6) else ("❌ edge 消失" if mean_net <= 0 else "⚠️ 方向在但不稳")
        print(f"  判定: {verdict}")
        return mean_net, np.mean(aucs_f), pos_win, len(nets)
    return float("nan"), np.mean(aucs_f), 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2025", "2026"], default=None,
                        help="只跑指定年份（分年防 swap）；不指定则两年连跑")
    args = parser.parse_args()

    results = {}
    for year in ([args.year] if args.year else ["2025", "2026"]):
        min_df = add_atr(load_minute_year(year))
        print(f"{year}: {len(min_df)} 分钟（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
        label = f"{year} 年（{'跨年验证组' if year == '2025' else '原样本对照'}）"
        r = run_wf(min_df, label)
        results[year] = {"net_bps": r[0] * 1e4, "auc": r[1], "pos_windows": r[2], "n_windows": r[3]}
        del min_df

    # 汇总落盘 + 打印
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if "2025" in results and "2026" in results:
        print("\n" + "=" * 60)
        print(f"跨年结论: 2025 净 {results['2025']['net_bps']:+.1f} bps/笔 vs 2026 净 {results['2026']['net_bps']:+.1f} bps/笔")
        print(f"          2025 AUC {results['2025']['auc']:.3f} vs 2026 AUC {results['2026']['auc']:.3f}")
        if results["2025"]["net_bps"] > 0:
            print("→ mbo 口径 edge 跨年成立，付费订阅（~$1,780/月）进入可讨论区间")
        else:
            print("→ 2025 年 edge 消失：26 天样本的 +30bps 是特例，mbo 方向止损（省 $21k/年）")


if __name__ == "__main__":
    main()
