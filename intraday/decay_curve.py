#!/usr/bin/env python3
"""信号衰减曲线：edge 存活窗口诊断（回测方法改进实验 3）。

问题：AUC 0.603 说的是「6 分钟窗口平均方向判断力」，但没回答「判断力在窗口内如何衰减」。
下一分钟开盘进场即归零（-4.8bps）暗示信号可能是瞬时的——若是，正确的执行方式是被动挂单
（做市型）而非吃延迟的主动成交。

设计：对 H = 1..6 分钟，各自做「不重叠采样 + walk-forward + 同超参数」训练评估，
输出两条曲线：
  ① AUC(H)——方向判断力 vs 窗口长度
  ② 净收益(H)——t 收盘进场、持有 H 分钟的口径（上限参考）
两者随 H 的形态直接回答「信号能等多久」。

注意：这是诊断实验（每个 H 独立训练同结构模型），不是调参——H 仍是 6 分钟策略的诊断扫描。

用法（磁盘 IO 病态期：单 H 单进程跑，防内存叠加）:
  python3 decay_curve.py --year 2025 --h-min 1 --h-max 3
  python3 decay_curve.py --year 2025 --h-min 4 --h-max 6
非连续 H 用 --h-list（如长锚点）：
  python3 decay_curve.py --year 2025 --h-list 40,50,60
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, PARAMS

OUT = Path(__file__).parent.parent / "tmp" / "decay_curve.json"


def run_H(min_df, H, train_days, test_days):
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    slices = window_slices(len(X), H, train_days, test_days)
    close_all = min_df["close"].values
    aucs, nets, n_tr = [], [], 0
    for tr, te, wid in slices:
        if len(set(y[tr])) < 2:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        aucs.append(auc)
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        trades = []
        pos, pos_end_min, t0 = 0, -1, 0
        te_e, te_x = e_idx[te], x_idx[te]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                pos_end_min = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end_min:
                exit_p = close_all[pos_end_min]
                e1 = close_all[t0]
                trades.append(pos * (exit_p - e1) / e1 - 3e-4 * (e1 + exit_p) / e1)
                pos = 0
        n_tr += len(trades)
        if trades:
            nets.append(np.mean(trades))
    aucs_f = [a for a in aucs if np.isfinite(a)]
    return {
        "H": H, "n_windows": len(slices), "n_trades": n_tr,
        "auc_mean": float(np.mean(aucs_f)),
        "auc_pos_windows": int(sum(1 for a in aucs_f if a > 0.52)),
        "net_bps": float(np.mean(nets) * 1e4) if nets else float("nan"),
        "net_pos_windows": int(sum(1 for s in nets if s > 0)),
        "n_nets": len(nets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2025")
    parser.add_argument("--h-min", type=int, default=1)
    parser.add_argument("--h-max", type=int, default=3)
    parser.add_argument("--h-list", default=None,
                        help="逗号分隔的 H 列表（如 40,50,60），优先于 --h-min/--h-max")
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=2)
    args = parser.parse_args()

    hs = [int(x) for x in args.h_list.split(",")] if args.h_list else list(range(args.h_min, args.h_max + 1))

    min_df = add_atr(load_minute_year(args.year))
    print(f"{args.year}: {len(min_df)} 分钟", flush=True)

    # 逐 H 落盘（支持分段跑合并 + 中断不丢已完成结果）
    OUT.parent.mkdir(exist_ok=True)
    merged = json.loads(OUT.read_text()) if OUT.exists() else {}
    for H in hs:
        r = run_H(min_df, H, args.train_days, args.test_days)
        merged[str(H)] = r
        OUT.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        print(f"H={H}: AUC {r['auc_mean']:.3f}（{r['auc_pos_windows']}/{r['n_windows']} 窗口）| "
              f"净 {r['net_bps']:+.1f} bps（{r['net_pos_windows']}/{r['n_nets']} 正）| 交易 {r['n_trades']}", flush=True)
    print(f"\n已写 {OUT}（累计 {len(merged)} 个 H）")


if __name__ == "__main__":
    main()
