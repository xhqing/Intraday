#!/usr/bin/env python3
"""测试脚本：载入 train.py 固化的模型文件测试（训练与测试解耦）。

载入 models/wf_H<H>/window_<N>.txt（LightGBM 原生格式），对每个窗口的 test 段做
一次一仓 + 持仓 H 分钟 + 止损 R 口径回测（与原 walk_forward_r.py 口径一致），汇总输出。

用法：
  python3 test.py --H 6 --stop-atr 2          # 载入 H=6 模型，止损 2×ATR
  python3 test.py --H 6 --stop-atr 10         # 同一批模型，不同止损对比
  python3 test.py --H 6 --stop-atr 0          # 无止损
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common_wf import load_minute, add_atr, build_nonoap, window_slices

MODELS_ROOT = Path(__file__).parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(prog="test.py", description="载入固化模型测试（walk-forward + 止损 R 口径）")
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--stop-atr", type=float, default=2.0, help="止损 N×ATR（0=无止损）")
    parser.add_argument("--price-only", action="store_true", help="只用价格特征（与训练时一致）")
    args = parser.parse_args()
    H, stop_atr = args.H, args.stop_atr

    # 1. 载入模型元数据
    suffix = "_priceonly" if args.price_only else ""
    model_dir = MODELS_ROOT / f"wf_H{H}{suffix}"
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"模型不存在：{model_dir}。先训练：python3 train.py --H {H} ...{' --price-only' if args.price_only else ''}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"载入模型：{model_dir}（{meta['n_windows']} 窗口，训练于 train_days={meta['train_days']}）", flush=True)

    # 2. 重建数据（与训练时同口径：同采样、同窗口切分）
    min_df = add_atr(load_minute())
    X, y, e_idx, x_idx = build_nonoap(min_df, H, price_only=args.price_only)
    if len(X) != meta["n_samples"]:
        raise SystemExit(f"数据样本数不符：当前 {len(X)} ≠ 训练时 {meta['n_samples']}（数据变了需重训）")
    close_all = min_df["close"].values
    atr_all = min_df["atr"].values
    high_all = min_df["high"].values
    low_all = min_df["low"].values
    print(f"数据重建: {len(X)} 样本（与训练时一致）", flush=True)

    # 3. 逐窗口：载入模型 → test 回测（一次一仓 + 止损 R 口径）
    slices = window_slices(len(X), H, meta["train_days"], meta["test_days"])
    results = []
    for tr, te, wid in slices:
        w = next(w for w in meta["windows"] if w["window"] == wid)
        if w["test"] != [te.start, te.stop]:
            raise SystemExit(f"窗口 {wid} 切分不一致（数据顺序变了？），需重训")
        booster = lgb.Booster(model_file=str(model_dir / f"window_{wid}.txt"))
        proba = booster.predict(X[te])
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))

        trades = []
        pos, entry_p, risk, stop_p, pos_end_min, t0 = 0, 0.0, 0.0, 0.0, -1, 0
        te_e, te_x = e_idx[te.start:te.stop], x_idx[te.start:te.stop]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                entry_p = close_all[t0]
                if stop_atr > 0:
                    risk = stop_atr * atr_all[t0]
                    stop_p = entry_p - pos * risk
                else:
                    # 无止损（用户定义）：双向 risk = entry price（做多止损在 0 / 做空止损在 2×entry，
                    # 即价格涨 100% 处，对称）——R 退化为以 entry 归一的收益率，与收益率口径严格相等
                    risk = entry_p
                    stop_p = float("nan")   # 无止损价（不触发止损检查）
                pos_end_min = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end_min:
                exited = False
                if stop_atr > 0:   # 只有设了止损才逐分钟检查
                    for t in range(t0 + 1, pos_end_min):
                        if pos > 0 and low_all[t] <= stop_p:
                            trades.append(((stop_p - entry_p) / risk, "stop"))
                            exited = True; pos = 0; break
                        elif pos < 0 and high_all[t] >= stop_p:
                            trades.append(((entry_p - stop_p) / risk, "stop"))
                            exited = True; pos = 0; break
                if not exited:
                    exit_p = close_all[pos_end_min]
                    gross_r = pos * (exit_p - entry_p) / risk
                    cost_r = 3e-4 * (entry_p + exit_p) / risk
                    trades.append((gross_r - cost_r, "time"))
                    pos = 0
        if trades:
            net = np.array([t[0] for t in trades if np.isfinite(t[0])])
            net_mean = net.mean() if len(net) else float("nan")
            n_stop = sum(1 for t in trades if t[1] == "stop")
        else:
            net_mean, n_stop = float("nan"), 0
        results.append((wid, auc, net_mean, len(trades), n_stop))
        print(f"窗口{wid}: AUC {auc:.3f}  净R {net_mean:+.3f}  交易{len(trades)} 止损{n_stop}", flush=True)

    # 4. 汇总
    aucs = [r[1] for r in results if np.isfinite(r[1])]
    nets = [r[2] for r in results if np.isfinite(r[2])]
    print(f"\n=== walk-forward 汇总（{len(results)} 窗口，H={H}，止损 {stop_atr}×ATR）===")
    print(f"  AUC: 均值 {np.mean(aucs):.3f} | 正窗口(>0.52) {sum(1 for a in aucs if a>0.52)}/{len(aucs)}")
    if nets:
        print(f"  净R: 均值 {np.mean(nets):+.3f} | 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)}")
        print(f"  {'✅ 稳健' if np.mean(nets) > 0 and sum(1 for s in nets if s>0) >= len(nets)*0.6 else '❌ 不稳定'}")


if __name__ == "__main__":
    main()
