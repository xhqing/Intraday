#!/usr/bin/env python3
"""拉取 QQQ mbo 全量订单簿多天数据（分段落盘、断点续传、断网可重跑续传）。

用法（代理环境）：
  zsh -c 'source ~/.zshrc && python3 fetch_mbo.py --days 20'
输出：data_tick/mbo_days/YYYY-MM-DD/ 每段 parquet
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent / "data_tick" / "mbo_days"


def fetch_day(h, day: str):
    """拉取一天的 mbo，按 2 小时分段落盘（断点续传）。"""
    out = OUT_DIR / day
    out.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(day, tz="UTC")
    end_ts = start_ts + pd.Timedelta(days=1)
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(hours=1), end_ts)   # 1 小时一段（代理下更稳）
        chunk = out / f"chunk_{cur:%H}.parquet"
        if chunk.exists() and chunk.stat().st_size > 1024:   # 有实际数据才跳过（防空段误判）
            cur = nxt
            continue
        ok = False
        for attempt in range(5):   # 5 次重试（代理慢，易断）
            try:
                store = h.timeseries.get_range(
                    dataset="XNAS.ITCH", schema="mbo",
                    start=cur, end=nxt, symbols=["QQQ"],
                )
                df = store.to_df()
                df.to_parquet(chunk)
                ok = True
                print(f"  {day} {cur:%H} 段 {len(df)} 条", flush=True)
                break
            except Exception as e:
                print(f"  {day} {cur:%H} 段失败({attempt+1}): {str(e)[:70]}", flush=True)
        if not ok:
            print(f"  ⚠️ {day} {cur:%H} 段跳过", flush=True)
        cur = nxt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（指定后忽略 --days/--end，拉 start ~ end 全区间）")
    args = parser.parse_args()

    # key 必须在 Historical() 之前设：取 zshrc 里 db- 开头的最后一条（真实 key，避免占位符"你的key"）
    import os
    import re
    keys = re.findall(r'DATABENTO_API_KEY="([^"]+)"', (Path.home() / ".zshrc").read_text())
    real = [k for k in keys if k.startswith("db-")][-1]
    os.environ["DATABENTO_API_KEY"] = real
    # 清代理直连（独立进程不 source zshrc，无代理环境变量；实测当前网络直连可用）
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(k, None)

    from databento import Historical
    h = Historical()
    if args.start:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        days = [d.isoformat() for d in pd.date_range(args.start, end, freq="D")]
    else:
        end = datetime.now(timezone.utc).date() if not args.end else datetime.strptime(args.end, "%Y-%m-%d").date()
        days = [(end - timedelta(days=i)).isoformat() for i in range(args.days - 1, -1, -1)]
    print(f"拉取 {len(days)} 天 QQQ mbo（{days[0]} ~ {days[-1]}），每 1 小时分段落盘断点续传")
    for d in days:
        fetch_day(h, d)


if __name__ == "__main__":
    main()
