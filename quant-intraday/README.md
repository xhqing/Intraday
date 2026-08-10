# quant-intraday — 日内策略探索（研究中的方向）

> 与已交付的日 K 策略（`quant-swing/`）分开的**日内策略探索目录**。当前处于研究阶段，结论尚未定型，**不用于实盘**。

## 目录内容

| 文件 | 用途 |
|---|---|
| `ml_orderflow_v2.py` | 订单流+OHLCV 特征 ML（正式版：清洗异常价、分天流式聚合防 OOM） |
| `ml_nonoap.py` | **不重叠采样 + 一次一仓 + 持仓 H 分钟**（当前最新，H 扫描） |
| `walk_forward.py` | walk-forward 滚动验证（重叠采样版，需按 ml_nonoap 修正） |
| `backtest_ml.py` | ML 信号完整回测（状态机一次一仓） |
| `fetch_mbo.py` | 拉取 mbo 订单流数据（分段落盘、断点续传） |
| `analyze_tick.py` / `analyze_mbo.py` / `analyze_flow.py` | 数据 / 订单流 OFI / 资金流分析 |
| `ml_trend.py` | 早期 15m OHLCV 特征 ML（AUC 0.52 无预测力，历史记录） |
| `data_tick/` | mbo 全量订单簿数据（26 个交易日 QQQ，15.7GB，已备份 GitHub） |

## 当前研究状态（2026-08-09）

- **信号预测力**：订单流+ML 对 5-6 分钟方向有预测力（AUC ~0.60）
- **关键修正**：不重叠采样 + 一次一仓 + 持仓 H 分钟，H=6 时净收益 +48 bps/笔（单次切分）
- **待完成**：H=6 的 walk-forward 验证（多窗口确认稳健性）
- 数据备份：GitHub Release `tick-full-2026-08-08` / `tick-full-2026-08-06`

## 数据说明

- 数据源：Databento XNAS.ITCH（QQQ mbo 全量订单簿）
- 免费额度已用大部分，后续拉取需注意额度
- `data_tick/` 已 gitignore（数据不进 git，备份在 GitHub Release）
