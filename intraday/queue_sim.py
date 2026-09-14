#!/usr/bin/env python3
"""队列优先级 fill rate 模拟：limit_sim 触及口径的「真实队列」订正（回测方法改进实验 5）。

背景：limit_sim 的 touch/strict 口径只验证「价格到过挂单价」，是 fill 的乐观上界——
价格瞬间穿过时被动单大概率没轮到（STRATEGY.md 第 10 节，实盘化前最后一关）。
本实验用 mbo 的 order_id 级事件重放订单簿队列，模拟「我们的限价单」在挂单价位的排队
位置，按「队列前移 + 成交吃单」判定真实成交——回答队列优先级口径的 fill rate。

事件语义（XNAS.ITCH 本地实测 2025-06-02 全天重放平衡验证：深度平稳 / bid<ask / spread 1 tick）：
  A (price,size,side)：挂入 +size
  C (price,size,side)：从 (price,side) 移除 size（leave quantity；其中成交性移除与同 order
     同 ts 的 F 事件配对，撤单性移除无配对 F）
  F：成交明细（与配对 C 冗余，用于区分移除原因）；T：隐式成交不占簿；R：罕见忽略
  Nasdaq 无隔夜保留单 + 数据从当日 00:00 UTC 起 → 从当日数据起点重放即完整

队列模型（Cont et al. 标准队列假设）：
  - 挂单时刻 t_q 的队列前置量 Q0 = 该 (price, side) 在 t_q 的存量（A 累加 / C 累减）
  - 挂单窗口 [t_q, t_q+60s) 内该价位 C 事件按序处理：
      成交性移除 s：吃队列；s > q → fill（部分成交即按全成计，净单 << 档位深度）
      撤单性移除 s：q = max(0, q - s)（前移但不会吃到我们）
  - 两口径：
      fill_all：撤单让位 + 成交吃单（标准队列模型）
      fill_exec：撤单不让位、仅成交性移除（保守下界）
  - 忽略自身挂单对簿的影响（标准近似）

信号与出场与 limit_sim 完全同口径（同代码路径：load_minute_year + build_nonoap +
window_slices + LightGBM PARAMS + qcut；H=6，挂单 = t+1 分钟整分钟，跨日撤单）：
  出场 = t+6 分钟收盘（委托价口径 close）市价，成本 maker 0 + taker 3bps。

用法（分年跑，防 swap）：
  python3 queue_sim.py --stage signals --year 2025    # 阶段1：生成信号表（重跑 walk-forward，~30min）
  python3 queue_sim.py --stage sim --year 2025        # 阶段2：逐日队列模拟（~1-2min/天）
  python3 queue_sim.py --stage sim --year 2025 --days 5   # 小规模验证
  python3 queue_sim.py --stage report --year 2025     # 阶段3：对比 touch/strict / 汇总
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from common_wf import add_atr, build_nonoap, window_slices
from verify_2025 import load_minute_year, H, TRAIN_DAYS, TEST_DAYS, PARAMS

TMP = Path(__file__).parent.parent / "tmp"
MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
DELTAS = [0.0, 0.01, 0.02, 0.05, 0.10]   # 挂单偏移（美元），与 limit_sim 一致
TAKER = 3e-4                              # 出场单边成本 3bps
WIN_NS = 60_000_000_000                   # 挂单窗口：t+1 分钟整分钟


# ---------- 阶段 1：信号生成（与 limit_sim 同代码路径，导出信号明细） ----------

def gen_signals(year: str) -> None:
    min_df = add_atr(load_minute_year(year))
    print(f"{year}: {len(min_df)} 分钟", flush=True)
    X, y, e_idx, x_idx = build_nonoap(min_df, H)
    slices = window_slices(len(X), H, TRAIN_DAYS, TEST_DAYS)
    dates = min_df.index
    close_all = min_df["close"].values
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
            t0, tx = int(te_e[i]), int(te_x[i])
            t1 = tx - H + 1
            if dates[t1].date() != dates[t0].date():   # 挂单跨日：撤单（与 limit_sim 一致）
                continue
            rows.append({
                "date": str(dates[t0].date()),
                "pos": 1 if q[i] == 4 else -1,
                "t1_ns": int(dates[t1].value),         # 挂单分钟左端（= 信号分钟收盘时刻）
                "tx_ns": int(dates[tx].value),
                "close_t": float(close_all[t0]),
                "exit_p": float(close_all[tx]),        # t+6 分钟收盘（委托价口径）
            })
        if wid % 20 == 0:
            print(f"  窗口 {wid}/{len(slices)-1}，信号 {len(rows)} 个", flush=True)
    df = pd.DataFrame(rows)
    out = TMP / f"queue_signals_{year}.csv"
    df.to_csv(out, index=False)
    print(f"信号 {len(df)} 个 → {out}")


# ---------- 阶段 2：逐日队列模拟 ----------

def load_day_events(day_dir: Path) -> pd.DataFrame:
    """读当日全部 chunk，返回簿事件表（ts 升序）：ts / side / pi / size / is_C / is_exec。

    pi = price×100 整数化（tick 网格，避免 float 相等比较）；is_exec = C 事件配对同
    order 同 ts 的 F（成交性移除）。T/R 事件丢弃（不占簿）。
    """
    cols = ["action", "side", "price", "size", "order_id", "ts_event"]
    parts = []
    for f in sorted(day_dir.glob("*.parquet")):
        if f.stat().st_size <= 1024:
            continue
        try:
            parts.append(pq.read_table(f, columns=cols).to_pandas())
        except Exception as e:
            print(f"  ⚠️ 跳过 {f.name}: {str(e)[:60]}", flush=True)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["ts"] = df["ts_event"].astype("int64")
    df = df.sort_values("ts", kind="mergesort")
    fills = df[df["action"] == "F"]
    fill_keys = set(zip(fills["order_id"].values, fills["ts"].values))
    del fills
    book = df[df["action"].isin(["A", "C"])].copy()
    del df
    book["pi"] = np.round(book["price"].values * 100).astype("int64")
    book["is_C"] = book["action"].values == "C"
    book["is_exec"] = book["is_C"] & np.fromiter(
        ((o, t) in fill_keys for o, t in zip(book["order_id"].values, book["ts"].values)),
        dtype=bool, count=len(book),
    )
    return book[["ts", "side", "pi", "size", "is_C", "is_exec"]]


def _fill_one(q: int, ts, sz, ex, count_cancels: bool):
    """单口径队列判定：返回 (fill, fill_ts)。

    q 初始 = Q0；窗口内 C 事件按序：
      成交性移除 s：s > q → fill；否则 q -= s
      撤单性移除 s：count_cancels 时 q = max(0, q-s)，否则跳过
    """
    for t, s, e in zip(ts, sz, ex):
        if e:
            if s > q:
                return True, int(t)
            q -= s
        elif count_cancels:
            q = max(0, q - int(s))
    return False, 0


def _side_groups(side_all, pi_all, ts_all, size_all, is_C_all, side_k):
    """某侧按价位分组的净存量前缀表：{pi: (ts[], cum_net[])}。

    cum_net[i] = 前 i+1 个事件的 A(+)/C(-) 累计——任意时刻 t 的存量 =
    cum_net[searchsorted(ts, t, 'left') - 1]（无事件则 0）。
    """
    m = side_all == side_k
    pi_s, ts_s, sz_s, c_s = pi_all[m], ts_all[m], size_all[m], is_C_all[m]
    order = np.argsort(pi_s, kind="stable")
    pi_s, ts_s, sz_s, c_s = pi_s[order], ts_s[order], sz_s[order], c_s[order]
    uniq = np.unique(pi_s)
    sa = np.searchsorted(pi_s, uniq, side="left")
    sb = np.searchsorted(pi_s, uniq, side="right")
    grp = {}
    for p, a, b in zip(uniq, sa, sb):
        t = ts_s[a:b]
        delta = np.where(c_s[a:b], -sz_s[a:b], sz_s[a:b])
        grp[int(p)] = (t, np.cumsum(delta))
    return grp


def _depth(grp, pi, t_q):
    """价位 pi 在时刻 t_q 的存量。"""
    g = grp.get(int(pi))
    if g is None:
        return 0
    ts, cum = g
    i = int(np.searchsorted(ts, t_q, side="left"))
    return int(cum[i - 1]) if i > 0 else 0


def _opposite_best(grp, pi_from, pi_to, t_q, find_min=True):
    """在 [pi_from, pi_to] 内找存量 > 0 的最优价位（ask 取最小 / bid 取最大）。

    只扫 limit ± 5 tick：更远的对手价存量视为错价单忽略。返回价位或 None。
    """
    rng = range(int(pi_from), int(pi_to) + 1) if find_min else range(int(pi_to), int(pi_from) - 1, -1)
    for p in rng:
        if _depth(grp, p, t_q) > 0:
            return p
    return None


def sim_day(day_dir: Path, sigs: pd.DataFrame) -> list:
    """当日全部信号 × δ 网格的队列模拟，返回逐信号结果行。

    marketable 判定：挂单时刻对手侧在 limit ± 5 tick 内有存量 → 直接按对手最优价成交
    （close_t 是委托价口径，可能落在对手侧，此时限价单是吃单不排队）。
    """
    book = load_day_events(day_dir)
    if book.empty or len(sigs) == 0:
        return []
    ts_all = book["ts"].values
    side_all = book["side"].values
    pi_all = book["pi"].values
    size_all = book["size"].values.astype(np.int64)
    is_C_all = book["is_C"].values
    is_ex_all = book["is_exec"].values

    # 两侧全簿净存量前缀表（marketable 判定用）
    ask_grp = _side_groups(side_all, pi_all, ts_all, size_all, is_C_all, "A")
    bid_grp = _side_groups(side_all, pi_all, ts_all, size_all, is_C_all, "B")

    # 目标价位集合（信号 × δ），一次布尔过滤 + 按价位分组索引
    targets = sorted({int(round((s["close_t"] - d if s["pos"] == 1 else s["close_t"] + d) * 100))
                      for _, s in sigs.iterrows() for d in DELTAS})
    mask = np.isin(pi_all, targets)
    print(f"  {day_dir.name[:10]}: 簿事件 {len(book):,}，目标价位 {len(targets)}，命中 {mask.sum():,}", flush=True)
    if not mask.any():
        return []
    hit_pos = np.flatnonzero(mask)
    order = np.argsort(pi_all[hit_pos], kind="stable")
    hit_pos = hit_pos[order]
    hit_pi = pi_all[hit_pos]
    sa = np.searchsorted(hit_pi, targets, side="left")
    sb = np.searchsorted(hit_pi, targets, side="right")
    grp = {p: hit_pos[a:b] for p, a, b in zip(targets, sa, sb)}

    rows = []
    for _, s in sigs.iterrows():
        side_k = "B" if s["pos"] == 1 else "A"      # 我们挂单所在侧
        opp_k, opp_grp = ("A", ask_grp) if s["pos"] == 1 else ("B", bid_grp)
        t_q = int(s["t1_ns"])
        t_c = t_q + WIN_NS
        for d in DELTAS:
            lim = s["close_t"] - d if s["pos"] == 1 else s["close_t"] + d
            lim_i = int(round(lim * 100))
            # ① marketable：对手侧 ±5 tick 内有存量 → 按对手最优价立即成交
            if s["pos"] == 1:
                best_opp = _opposite_best(opp_grp, lim_i - 5, lim_i, t_q, find_min=True)
            else:
                best_opp = _opposite_best(opp_grp, lim_i, lim_i + 5, t_q, find_min=False)
            if best_opp is not None:
                px = best_opp / 100.0
                r = s["pos"] * (s["exit_p"] - px) / px - TAKER * s["exit_p"] / px
                rows.append({"date": s["date"], "pos": s["pos"], "delta": d, "limit": lim,
                             "Q0": -1, "marketable": True, "fill_all": True, "fill_exec": True,
                             "fill_ts": float(t_q), "r": r})
                continue
            # ② 排队：本侧队列模拟
            idx = grp.get(lim_i, np.array([], dtype=np.int64))
            m2 = side_all[idx] == side_k if len(idx) else idx
            idx = idx[m2] if len(idx) else idx
            if len(idx) == 0:
                rows.append({"date": s["date"], "pos": s["pos"], "delta": d, "limit": lim,
                             "Q0": 0, "marketable": False, "fill_all": False, "fill_exec": False, "fill_ts": 0.0})
                continue
            ev_ts = ts_all[idx]
            # Q0：挂单前净存量（A +size / C -size）
            i0 = int(np.searchsorted(ev_ts, t_q, side="left"))
            delta = np.where(is_C_all[idx][:i0], -size_all[idx][:i0], size_all[idx][:i0])
            q0 = int(delta.sum())
            # 窗口内 C 事件
            i1 = int(np.searchsorted(ev_ts, t_c, side="left"))
            w = np.flatnonzero(is_C_all[idx][i0:i1]) + i0
            wt = ev_ts[w]
            ws = size_all[idx][w]
            we = is_ex_all[idx][w]
            fill_a, ts_a = _fill_one(q0, wt, ws, we, True)
            fill_e, ts_e = _fill_one(q0, wt, ws, we, False)
            filled = fill_a or fill_e
            r = (s["pos"] * (s["exit_p"] - lim) / lim - TAKER * s["exit_p"] / lim) if filled else 0.0
            rows.append({"date": s["date"], "pos": s["pos"], "delta": d, "limit": lim,
                         "Q0": q0, "marketable": False,
                         "fill_all": fill_a, "fill_exec": fill_e,
                         "fill_ts": float(ts_a or ts_e), "r": r})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["signals", "sim", "report"], required=True)
    parser.add_argument("--year", default="2025")
    parser.add_argument("--days", type=int, default=None, help="sim 阶段只跑前 N 个交易日（小规模验证）")
    args = parser.parse_args()

    if args.stage == "signals":
        gen_signals(args.year)

    elif args.stage == "sim":
        sigs = pd.read_csv(TMP / f"queue_signals_{args.year}.csv")
        day_dirs = sorted(d for d in MBODIR.iterdir() if d.is_dir() and d.name.startswith(args.year))
        if args.days:
            day_dirs = day_dirs[: args.days]
        out_rows = []
        for day_dir in day_dirs:
            sigs_day = sigs[sigs["date"] == day_dir.name[:10]]
            if len(sigs_day) == 0:
                continue
            out_rows.extend(sim_day(day_dir, sigs_day))
        df = pd.DataFrame(out_rows)
        suffix = f"_{args.days}d" if args.days else ""
        out_csv = TMP / f"queue_sim_{args.year}{suffix}.csv"
        df.to_csv(out_csv, index=False)
        print(f"结果 {len(df)} 行 → {out_csv}")

    elif args.stage == "report":
        df = pd.read_csv(TMP / f"queue_sim_{args.year}.csv")
        print(f"\n=== {args.year} 队列优先级模拟（信号 {len(df)//len(DELTAS)} 个 × δ 网格）===")
        print(f"{'δ(美分)':>8} | {'marketable':>10} | {'fill_all':>9} | {'fill_exec':>9} | {'Q0中位':>8} | {'条件bps(all)':>12} | {'期望bps(all)':>12} | {'期望bps(exec)':>13}")
        for d in DELTAS:
            g = df[df["delta"] == d]
            mk = g["marketable"].mean() if "marketable" in g else float("nan")
            fa, fe = g["fill_all"].mean(), g["fill_exec"].mean()
            ca = g.loc[g["fill_all"], "r"].mean() * 1e4
            ea = g["r"].where(g["fill_all"], 0).mean() * 1e4
            ee = g["r"].where(g["fill_exec"], 0).mean() * 1e4
            q0m = g.loc[~g["marketable"], "Q0"].median() if "marketable" in g else g["Q0"].median()
            print(f"{d*100:>8.0f} | {mk*100:>9.1f}% | {fa*100:>8.1f}% | {fe*100:>8.1f}% | {q0m:>8.0f} | {ca:>+12.1f} | {ea:>+12.1f} | {ee:>+13.1f}")
        print("基准：touch 94.7% / strict 92.5%（limit_sim δ=0，触及口径乐观上界）")


if __name__ == "__main__":
    main()
