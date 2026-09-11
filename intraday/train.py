#!/usr/bin/env python3
"""训练脚本：滚动窗口训练模型并固化到文件（测试与训练解耦，test.py 载入模型测试）。

每个窗口训练一个 LightGBM 模型，保存为 LightGBM 原生格式 txt（跨版本稳定）：
  models/wf_H<H>/window_<N>.txt   模型文件
  models/wf_H<H>/meta.json        全部窗口元数据（窗口切分、超参数、训练样本范围）

用法：
  python3 train.py --H 6 --train-days 8 --test-days 2 --n-est 300
  # 训练完成后：python3 test.py --H 6 --stop-atr 2   # 载入模型测试
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb

from common_wf import load_minute, add_atr, build_nonoap, window_slices

MODELS_ROOT = Path(__file__).parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(prog="train.py", description="滚动窗口训练 + 固化模型")
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--n-est", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--price-only", action="store_true", help="只用价格特征（去 ofi/flow_large/nb/avg_size）")
    args = parser.parse_args()
    H = args.H

    min_df = add_atr(load_minute())
    print(f"分钟级: {len(min_df)}（{min_df.index.date.min()} ~ {min_df.index.date.max()}）", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H, price_only=args.price_only)
    print(f"不重叠样本: {len(X)}（H={H}）", flush=True)

    slices = window_slices(len(X), H, args.train_days, args.test_days)
    suffix = "_priceonly" if args.price_only else ""
    out_dir = MODELS_ROOT / f"wf_H{H}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "H": H, "train_days": args.train_days, "test_days": args.test_days,
        "n_est": args.n_est, "lr": args.lr, "leaves": args.leaves,
        "n_samples": len(X), "n_windows": len(slices), "price_only": args.price_only, "windows": [],
    }
    for tr, te, wid in slices:
        model = lgb.LGBMClassifier(n_estimators=args.n_est, learning_rate=args.lr,
                                   num_leaves=args.leaves, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, n_jobs=2)
        model.fit(X[tr], y[tr])
        model.booster_.save_model(str(out_dir / f"window_{wid}.txt"))
        meta["windows"].append({
            "window": wid,
            "train": [tr.start, tr.stop],
            "test": [te.start, te.stop],
            "entry_first_min": int(e_idx[te.start]),   # test 首样本的分钟索引（test.py 校验用）
        })
        print(f"窗口 {wid}: train [{tr.start}:{tr.stop}] → models/wf_H{H}/window_{wid}.txt", flush=True)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n完成：{len(slices)} 个模型已固化到 {out_dir}/（meta.json 含窗口切分信息）")
    extra = " --price-only" if args.price_only else ""
    print(f"测试：python3 test.py --H {H} --stop-atr 2{extra}")


if __name__ == "__main__":
    main()
