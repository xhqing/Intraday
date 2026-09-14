#!/usr/bin/env python3
"""深度订单簿特征构建：盘口状态 + 事件流微结构（深挖实验 1，2026-09-13 开启）。

背景：F-label 终极检验只关闭了现有 15 个基础特征（AUC 0.51）。未测的特征家族
（文献有据）：CKS 盘口 OFI（best 档深度变化，含撤单/成交对簿的影响——与现有
「A 事件加总版」ofi 构造不同）、多档深度不平衡、撤单流、成交流、事件微结构。
本脚本从 mbo 事件流逐日构建这些分钟级特征，落盘 parquet 缓存供 verify_deep 检验。

特征清单（分钟 t，全部只用 ≤ t 分钟结束时刻的信息）：
  簿截面（t 分钟结束时刻的盘口状态）：
    spread_ticks   (best_ask − best_bid) / tick
    depth_imb_1    (vB1 − vA1) / (vB1 + vA1)          买方压力为正
    depth_imb_5    (ΣvB5 − ΣvA5) / (ΣvB5 + ΣvA5)
    book_vol_5     ΣvB5 + ΣvA5（簿厚度量级）
    mid_ret_1 / mid_mom_5 / mid_mom_15   mid = (bb+ba)/2 的动量
    ofi_cks        分钟粒度 CK 调整差分 OFI：bb 上移 +vB1 / 持平 ΔvB1 / 下移 −vB1(prev)，
                   ask 对称取负，两者之和（近似逐 tick CKS，分钟聚合）
  事件流（t 分钟内）：
    add_imb        (ΣA_B − ΣA_A) / (ΣA_B + ΣA_A)      新增委托买方压力
    cancel_imb     (ΣC_A − ΣC_B) / (ΣC_A + ΣC_B)      撤单：卖方撤退多 = 正
    trade_imb      (taker_buy − taker_sell) / (Σ)     主动成交买方压力
                   taker_buy = F(side=A 被吃) + T(side=B)，taker_sell 对称
    n_add / n_cancel / n_fill / cr_ratio = n_cancel / max(n_add,1)
    avg_add_size / avg_fill_size

口径说明：
  - 分钟网格：每交易日 13:30~20:00 UTC（EDT 9:30~16:00，数据全在夏令时）390 分钟
  - 簿状态取分钟右端（含整分钟演化）；事件按分钟左端桶聚合
  - 错价过滤：价位偏离当日 F 成交中位价 ±1% 之外视为错价、深度记 0
  - 重放负深度（数据边界伪影）clamp 为 0
  - 全天重放从当日 00:00 UTC 起（Nasdaq 无隔夜单，经 queue_sim 验证）

用法：
  python3 deep_features.py --year 2025 [--days N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TMP = Path(__file__).parent.parent / "tmp"
MBODIR = Path(__file__).parent / "data_tick" / "mbo_days"
TICK = 0.01
N_MIN = 390   # 13:30~20:00 UTC


def load_day(day_dir: Path):
    """读当日全部事件 → (book_df, trade_df)。book=A/C（簿），trade=F/T（成交）。"""
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
        return None, None
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["ts"] = df["ts_event"].astype("int64")
    df = df.sort_values("ts", kind="mergesort")
    book = df[df["action"].isin(["A", "C"])].copy()
    trade = df[df["action"].isin(["F", "T"])].copy()
    del df
    book["pi"] = np.round(book["price"].values * 100).astype("int64")
    return book, trade


def book_matrix(side_df: pd.DataFrame, boundaries: np.ndarray) -> tuple:
    """某侧簿事件 → (价位数组升序, 深度矩阵 [n_px, n_min])。"""
    if len(side_df) == 0:
        return np.array([]), np.zeros((0, len(boundaries)))
    pi = side_df["pi"].values
    ts = side_df["ts"].values
    sz = side_df["size"].values.astype(np.int64)
    is_c = (side_df["action"].values == "C")
    order = np.argsort(pi, kind="stable")
    pi, ts, sz, is_c = pi[order], ts[order], sz[order], is_c[order]
    uniq = np.unique(pi)
    sa = np.searchsorted(pi, uniq, side="left")
    sb = np.searchsorted(pi, uniq, side="right")
    mat = np.zeros((len(uniq), len(boundaries)), dtype=np.int64)
    for k, (a, b) in enumerate(zip(sa, sb)):
        t = ts[a:b]
        delta = np.where(is_c[a:b], -sz[a:b], sz[a:b])
        cum = np.cumsum(delta)
        idx = np.searchsorted(t, boundaries, side="left")
        mat[k] = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0)
    return uniq, mat


def top_levels(pi: np.ndarray, mat: np.ndarray, is_bid: bool, n_top: int = 5):
    """每分钟边界的 best 价与前 n_top 档深度（bid 取最高 / ask 取最低，跳过零深度档）。

    返回 best_px (n_min,), vol_k (k, n_min) 各档深度。
    """
    order = np.argsort(-pi if is_bid else pi, kind="stable")
    px_s, d_s = pi[order], mat[order]
    pos = d_s > 0
    first = np.argmax(pos, axis=0)                     # 每列第一个非零档行号
    any_nz = pos.any(axis=0)
    first = np.where(any_nz, first, 0)
    n_min = mat.shape[1]
    best = np.where(any_nz, px_s[first], 0.0)
    vols = np.zeros((n_top, n_min))
    rows = first[:, None] + np.arange(n_top)[None, :]  # (n_min, n_top)
    rows = np.minimum(rows, len(px_s) - 1)
    for k in range(n_top):
        v = d_s[rows[:, k], np.arange(n_min)]
        vols[k] = np.where(any_nz & (d_s[rows[:, k], np.arange(n_min)] > 0),
                           d_s[rows[:, k], np.arange(n_min)], 0)
    return best, vols


def build_day(day_dir: Path) -> pd.DataFrame | None:
    book, trade = load_day(day_dir)
    if book is None or len(book) == 0:
        return None
    # 分钟网格（该日 00:00 UTC 为锚）
    day0 = int(pd.Timestamp(day_dir.name[:10], tz="UTC").value)
    m0 = day0 + (13 * 60 + 30) * 60_000_000_000        # 13:30 UTC 左端
    lefts = m0 + np.arange(N_MIN) * 60_000_000_000
    rights = lefts + 60_000_000_000                    # 边界（右端）

    # 成交中位价（错价过滤基准）
    t_px = trade["price"].values if trade is not None and len(trade) else np.array([])
    med = np.median(t_px) if len(t_px) else float("nan")

    # 两侧簿矩阵
    _, mat_b = book_matrix(book[book["side"] == "B"], rights)
    pi_a_all, mat_a = book_matrix(book[book["side"] == "A"], rights)
    pi_b_all = book_matrix(book[book["side"] == "B"], rights)[0]
    # 错价过滤（±1%）
    if np.isfinite(med):
        lo, hi = med * 0.99, med * 1.01
        mat_b[~((pi_b_all >= lo * 100) & (pi_b_all <= hi * 100))] = 0
        mat_a[~((pi_a_all >= lo * 100) & (pi_a_all <= hi * 100))] = 0
    mat_b = np.maximum(mat_b, 0)
    mat_a = np.maximum(mat_a, 0)

    bb, vB = top_levels(pi_b_all, mat_b, is_bid=True)
    ba, vA = top_levels(pi_a_all, mat_a, is_bid=False)
    valid = (bb > 0) & (ba > 0)
    bb, ba = bb / 100.0, ba / 100.0
    mid = (bb + ba) / 2
    vB1, vA1 = vB[0].astype(float), vA[0].astype(float)
    sB5, sA5 = vB.sum(axis=0).astype(float), vA.sum(axis=0).astype(float)

    # CKS 分钟 OFI（best 档调整差分）
    ofi_b = np.where(np.roll(bb, 1) < bb, vB1,                    # bb 上移
             np.where(np.roll(bb, 1) == bb, vB1 - np.roll(vB1, 1),  # 持平
                      -np.roll(vB1, 1)))                            # bb 下移
    ofi_a = np.where(np.roll(ba, 1) > ba, vA1,
             np.where(np.roll(ba, 1) == ba, -(vA1 - np.roll(vA1, 1)),
                      np.roll(vA1, 1)))
    ofi_a = -ofi_a   # ask 深度增加 = 卖压 = 负
    ofi_cks = ofi_b + ofi_a
    ofi_cks[0] = 0.0

    # 事件流统计（分钟左端桶）
    def _minute_bucket(ts_arr):
        return np.searchsorted(lefts, ts_arr, side="right") - 1

    out = pd.DataFrame({
        "minute_ns": lefts,
        "bb": bb, "ba": ba, "mid": mid, "valid": valid,
        "spread_ticks": (ba - bb) / TICK,
        "depth_imb_1": np.where(vB1 + vA1 > 0, (vB1 - vA1) / (vB1 + vA1), 0),
        "depth_imb_5": np.where(sB5 + sA5 > 0, (sB5 - sA5) / (sB5 + sA5), 0),
        "book_vol_5": sB5 + sA5,
        "ofi_cks": ofi_cks,
    })
    if trade is not None and len(trade):
        a_b = book[(book["action"] == "A") & (book["side"] == "B")]
        a_a = book[(book["action"] == "A") & (book["side"] == "A")]
        c_b = book[(book["action"] == "C") & (book["side"] == "B")]
        c_a = book[(book["action"] == "C") & (book["side"] == "A")]
        f = trade[trade["action"] == "F"]
        t = trade[trade["action"] == "T"]
        def _sum_side(d):
            if len(d) == 0:
                return np.zeros(N_MIN)
            b = _minute_bucket(d["ts"].values)
            m = (b >= 0) & (b < N_MIN)
            s = np.zeros(N_MIN)
            np.add.at(s, b[m], d["size"].values[m].astype(np.float64))
            return s
        def _cnt(d):
            if len(d) == 0:
                return np.zeros(N_MIN)
            b = _minute_bucket(d["ts"].values)
            m = (b >= 0) & (b < N_MIN)
            c = np.zeros(N_MIN)
            np.add.at(c, b[m], 1)
            return c
        sAB, sAA = _sum_side(a_b), _sum_side(a_a)
        sCB, sCA = _sum_side(c_b), _sum_side(c_a)
        nA_cnt, nC_cnt = _cnt(book[book["action"] == "A"]), _cnt(book[book["action"] == "C"])
        out["add_imb"] = np.where(sAB + sAA > 0, (sAB - sAA) / (sAB + sAA), 0)
        out["cancel_imb"] = np.where(sCB + sCA > 0, (sCA - sCB) / (sCB + sCA), 0)
        out["n_add"] = nA_cnt
        out["n_cancel"] = nC_cnt
        out["cr_ratio"] = nC_cnt / np.maximum(nA_cnt, 1)
        out["avg_add_size"] = np.where(nA_cnt > 0, (sAB + sAA) / np.maximum(nA_cnt, 1), 0)
        # 成交不平衡（逐笔聚合到分钟）：F 的 side=被动方（A 被吃 → taker 买），T 的 side=taker 方向（B → taker 买）
        tr_frames = []
        for d, is_f in ((f, True), (t, False)):
            if len(d):
                tr_frames.append(d[["ts", "side", "size"]].assign(is_f=is_f))
        if tr_frames:
            tr = pd.concat(tr_frames, ignore_index=True)
            b = _minute_bucket(tr["ts"].values)
            m = (b >= 0) & (b < N_MIN)
            is_buy = np.where(tr["is_f"].values, tr["side"].values == "A", tr["side"].values == "B")
            sz_v = tr["size"].values.astype(np.float64)
            tb = np.zeros(N_MIN); tsv = np.zeros(N_MIN); nf = np.zeros(N_MIN); vsz = np.zeros(N_MIN)
            np.add.at(tb, b[m & is_buy], sz_v[m & is_buy])
            np.add.at(tsv, b[m & ~is_buy], sz_v[m & ~is_buy])
            np.add.at(nf, b[m], 1)
            np.add.at(vsz, b[m], sz_v[m])
            out["trade_imb"] = np.where(tb + tsv > 0, (tb - tsv) / (tb + tsv), 0)
            out["n_fill"] = nf
            out["avg_fill_size"] = np.where(nf > 0, vsz / np.maximum(nf, 1), 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2025")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    day_dirs = sorted(d for d in MBODIR.iterdir() if d.is_dir() and d.name.startswith(args.year))
    if args.days:
        day_dirs = day_dirs[: args.days]
    frames = []
    for day_dir in day_dirs:
        df = build_day(day_dir)
        if df is None:
            continue
        frames.append(df)
        print(f"  {day_dir.name[:10]} ok（{df['valid'].sum()} 分钟有盘口）", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["ts"] = pd.to_datetime(all_df["minute_ns"], unit="ns", utc=True)
    # mid 动量（跨分钟；跨日边界用前一天尾分钟，与原管道 mom 同款近似）
    mid = all_df["mid"].where(all_df["valid"], np.nan).ffill()
    all_df["mid_ret_1"] = mid.pct_change()
    all_df["mid_mom_5"] = mid / mid.shift(5) - 1
    all_df["mid_mom_15"] = mid / mid.shift(15) - 1
    out = TMP / f"deep_feat_{args.year}.parquet"
    all_df.to_parquet(out, index=False)
    print(f"{len(all_df)} 分钟 × {len(all_df.columns)} 列 → {out}")


if __name__ == "__main__":
    main()
