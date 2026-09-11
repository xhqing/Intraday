#!/usr/bin/env python3
"""富途 1m 成交价口径全流程：不重叠采样 + 一次一仓 + 持仓 H 分钟 + walk-forward。

背景：mbo 聚合分钟数据是「委托簿口径」（挂单价/委托量），与实盘可得的富途 1m K（成交价口径）
特征相关系数 ≈0（2026-08-16 实测）——此前 mbo 实验的结论不能直接迁移实盘。本脚本用
富途 QQQ 1m（2021 起 5.5 年，成交价）重做：特征重算 → 滚动窗口训练 → test 段一次一仓回测。

与 train/test 解耦版同构：--do train 固化模型，--do test 载入测试（模型存 models/futu_1m_H<H>/）。

用法：
  python3 futu_wf.py --do train --H 6
  python3 futu_wf.py --do test --H 6 --stop-atr 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from ml_orderflow_v2 import build_features

FUTU_1M = Path(__file__).parent.parent / ".claude" / "skills" / "quant" / "data_cache" / "US.QQQ" / "1m.parquet"
MODELS_ROOT = Path(__file__).parent / "models"


def load_futu_minute() -> pd.DataFrame:
    """富途 1m → 与 mbo 聚合版同结构的分钟 df（vol=成交量；订单流列置 0 由 price-only 剔除）。"""
    df = pd.read_parquet(FUTU_1M)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"volume": "vol"})
    df["ofi"] = 0.0; df["flow_large"] = 0.0
    df["n_buy"] = 0; df["n_sell"] = 0; df["avg_size"] = 0.0
    return df[["open", "high", "low", "close", "vol", "ofi", "flow_large", "n_buy", "n_sell", "avg_size"]]


def add_atr(min_df, period=14):
    prev_close = min_df["close"].shift(1)
    tr = pd.concat([
        min_df["high"] - min_df["low"],
        (min_df["high"] - prev_close).abs(),
        (min_df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    min_df["atr"] = tr.rolling(period).mean()
    return min_df


def build_nonoap(min_df, H):
    feats = build_features(min_df, 60).drop(columns=["ofi", "flow_large", "nb", "avg_size"], errors="ignore")
    close = min_df["close"]
    n = len(min_df)
    X_list, y_list, e_idx, x_idx = [], [], [], []
    for t in range(0, n - H, H):
        if feats.iloc[t].isna().all():
            continue
        fr = close.iloc[t + H] / close.iloc[t] - 1
        if not np.isfinite(fr):
            continue
        X_list.append(feats.iloc[t].values)
        y_list.append(1 if fr > 0 else 0)
        e_idx.append(t)
        x_idx.append(t + H)
    return np.array(X_list), np.array(y_list), np.array(e_idx), np.array(x_idx)


def do_train(H, train_days, test_days, n_est):
    min_df = add_atr(load_futu_minute())
    print(f"富途 1m: {len(min_df)}（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    print(f"不重叠样本: {len(X)}（H={H}）", flush=True)

    spd = 390 // H
    train_n, test_n = train_days * spd, test_days * spd
    out_dir = MODELS_ROOT / f"futu_1m_H{H}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = []
    start = 0
    wid = 0
    while start + train_n + test_n <= len(X):
        slices.append((slice(start, start + train_n), slice(start + train_n, start + train_n + test_n), wid))
        start += test_n
        wid += 1
    meta = {"H": H, "train_days": train_days, "test_days": test_days, "n_est": n_est,
            "n_samples": len(X), "n_windows": len(slices), "source": "futu_1m", "windows": []}
    for tr, te, wid in slices:
        model = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
        model.fit(X[tr], y[tr])
        model.booster_.save_model(str(out_dir / f"window_{wid}.txt"))
        meta["windows"].append({"window": wid, "train": [tr.start, tr.stop], "test": [te.start, te.stop]})
        if wid % 50 == 0 or wid == len(slices) - 1:
            print(f"  窗口 {wid}/{len(slices)-1}", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"完成：{len(slices)} 模型 → {out_dir}/")


def do_test(H, stop_atr):
    model_dir = MODELS_ROOT / f"futu_1m_H{H}"
    meta = json.loads((model_dir / "meta.json").read_text())
    min_df = add_atr(load_futu_minute())
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    assert len(X) == meta["n_samples"], "数据漂移，需重训"
    close_all = min_df["close"].values
    atr_all = min_df["atr"].values
    high_all = min_df["high"].values
    low_all = min_df["low"].values

    results = []
    for w in meta["windows"]:
        wid = w["window"]
        te = slice(*w["test"])
        booster = lgb.Booster(model_file=str(model_dir / f"window_{wid}.txt"))
        proba = booster.predict(X[te])
        auc = roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else float("nan")
        q = np.array(pd.qcut(proba, 5, labels=False, duplicates="drop"))
        trades = []
        pos, entry_p, risk, stop_p, pos_end_min, t0 = 0, 0.0, 0.0, 0.0, -1, 0
        te_e, te_x = e_idx[te], x_idx[te]
        for i in range(len(te_e)):
            if pos == 0 and q[i] in (0, 4):
                pos = 1 if q[i] == 4 else -1
                t0 = te_e[i]
                entry_p = close_all[t0]
                if stop_atr > 0:
                    risk = stop_atr * atr_all[t0]
                    stop_p = entry_p - pos * risk
                else:
                    risk = entry_p   # 双向 risk=entry（收益率口径）
                    stop_p = float("nan")
                pos_end_min = te_x[i]
            elif pos != 0 and te_e[i] >= pos_end_min:
                exited = False
                if stop_atr > 0:
                    for t in range(t0 + 1, pos_end_min):
                        if pos > 0 and low_all[t] <= stop_p:
                            trades.append(((stop_p - entry_p) / risk, "stop")); exited = True; pos = 0; break
                        elif pos < 0 and high_all[t] >= stop_p:
                            trades.append(((entry_p - stop_p) / risk, "stop")); exited = True; pos = 0; break
                if not exited:
                    exit_p = close_all[pos_end_min]
                    gross_r = pos * (exit_p - entry_p) / risk
                    cost_r = 3e-4 * (entry_p + exit_p) / risk
                    trades.append((gross_r - cost_r, "time"))
                    pos = 0
        net = np.array([t[0] for t in trades if np.isfinite(t[0])]) if trades else np.array([])
        results.append((wid, auc, net.mean() if len(net) else float("nan"), len(trades)))
    aucs = [r[1] for r in results if np.isfinite(r[1])]
    nets = [r[2] for r in results if np.isfinite(r[2])]
    print(f"\n=== 富途 1m walk-forward 汇总（{len(results)} 窗口，H={H}，止损 {stop_atr}×ATR，5.5 年）===")
    print(f"  AUC: 均值 {np.mean(aucs):.3f} | 正窗口(>0.52) {sum(1 for a in aucs if a>0.52)}/{len(aucs)}")
    if nets:
        print(f"  净R: 均值 {np.mean(nets):+.4f}（{np.mean(nets)*1e4:+.1f} bps/笔）| 正窗口 {sum(1 for s in nets if s>0)}/{len(nets)}")
        print(f"  {'✅ 稳健' if np.mean(nets) > 0 and sum(1 for s in nets if s>0) >= len(nets)*0.6 else '❌ 不稳定'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do", choices=["train", "test"], required=True)
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--n-est", type=int, default=300)
    parser.add_argument("--stop-atr", type=float, default=0.0)
    args = parser.parse_args()
    if args.do == "train":
        do_train(args.H, args.train_days, args.test_days, args.n_est)
    else:
        do_test(args.H, args.stop_atr)
