# intraday — 日内策略探索（研究中的方向）

> 与已交付的日 K 策略（`swing/`）分开的**日内策略探索目录**。当前处于研究阶段，结论尚未定型，**不用于实盘**。

## 目录内容

| 文件 | 用途 |
|---|---|
| **`train.py`** | **训练**：滚动窗口训练 + 模型固化（`models/wf_H<H>/`，含 `--price-only` 开关） |
| **`test.py`** | **测试**：载入固化模型 walk-forward 测试（改止损无需重训；`--stop-atr 0` = 双向 risk=entry 口径） |
| **`common_wf.py`** | 共用：数据加载 / ATR / 不重叠采样 / 窗口切分（训练测试同口径） |
| `verify_2025.py` | **跨年验证**：2025 年（92 交易日）同口径 walk-forward，检验 edge 是否跨年成立（`--year` 分年跑防 swap） |
| `decay_curve.py` | **信号衰减曲线**：H=1..6 各自独立 walk-forward，AUC(H)/净收益(H) 定位 edge 存活窗口 |
| `limit_sim.py` | **限价单模拟**：t+1 分钟挂限价单（δ 网格）被动成交 vs 市价对比（fill rate / 条件收益 / 每信号期望） |
| `futu_wf.py` | 富途 1m 成交价口径全流程（train/test）——5.5 年验证：成交价口径无 edge（证伪） |
| `ml_orderflow_v2.py` | 订单流+OHLCV 特征与分钟聚合（清洗异常价、分天防 OOM；被 common_wf 复用） |
| `ml_nonoap.py` | 不重叠采样 + 一次一仓 + H 扫描（H=6 最优的来源实验） |
| `walk_forward_r.py` / `walk_forward_nonoap.py` / `walk_forward.py` | walk-forward 一体版（旧，已被 train/test 解耦取代；r 版含止损 R 口径） |
| `backtest_ml.py` | ML 信号完整回测（状态机一次一仓，历史探索版） |
| `fetch_mbo.py` | 拉取 mbo 订单流数据（分段落盘、断点续传） |
| `analyze_tick.py` / `analyze_mbo.py` / `analyze_flow.py` | 数据 / 订单流 OFI / 资金流分析 |
| `ml_trend.py` | 早期 15m OHLCV 特征 ML（AUC 0.52 无预测力——15m 预测 3~24 小时无 edge，与分钟级 6 分钟有 edge 不矛盾：时间尺度不同） |
| `walk_forward_experiment.ipynb` | walk-forward 实验笔记本（分块调试用） |
| `data_tick/` | mbo 全量订单簿数据（26 个交易日 QQQ，15.7GB，已备份 GitHub） |
| `models/` | 固化模型（gitignore，train.py 可重生） |

## 当前研究状态（2026-08-30 限价单模拟版）

- **结论（2026-08-30，「不可交易」被推翻）**：2026-08-29 曾判定「edge 存活窗口短于一分钟、不可交易（t+1 市价进场 −4.8bps）、Databento Plus 不予考虑」——限价单模拟实验推翻该判定的「不可交易」部分：**edge 怕延迟但不怕被动等**。t+1 分钟内以信号价挂限价单（δ=0，maker 进 0 + taker 出 3bps），2025 年 5090 信号：**fill 92.5%（strict 穿过半 tick 口径）、成交笔均 +25.3bps、每信号期望 +23.4bps**——与回测口径 +24.0bps 同量级，而同位置主动市价 −4.8bps。方向对称（多 +24.0 / 空 +26.6）、四个月全正（+29.0 / +25.8 / +27.5 / +12.3）、δ 越深期望越低（挂当前价不追价最优）。**保留：fill 是分钟触及口径的乐观上界，真实队列优先级 fill rate 待 order_id 级模拟**
- **结论（订正存档，2026-08-29）**：2025 年跨年验证原报「+64.7 bps/笔」经逐笔明细复核订正——① 坏点（盘后 close=0 伪影 856 分钟）31 笔假交易虚增 43bps；② 剔坏点后 **+24.0 bps/笔（AUC 0.603、88/98 窗口正、3596 笔），2026 年修正版 +39.2 bps（AUC 0.589、18/20 正）**——方向预测力跨年稳健成立；③ t+1 开盘市价进场 −4.8bps（每月皆负）——主动吃延迟拿不到 edge
- **信号衰减曲线（2026-08-30，`decay_curve.py`）**：H=1..6 各自独立 walk-forward——AUC 单调衰减 0.638→0.603（判断力集中在前 1 分钟）、每笔净收益平坦 +25.7~+21.5bps（多持有不多赚）、笔数与 H 成反比（H=1 是 H=6 的 6 倍）。edge 在信号后第 1 分钟内兑现，之后是噪声
- **口径结论（2026-08-16~17）**：mbo 聚合分钟（委托簿口径）与富途 1m（成交价口径）特征相关 ≈0，两个口径是不同的信息世界；富途成交价口径 5.5 年重训验证无 edge（−6bps/笔，futu_wf.py 819 窗口）
- **实盘数据成本（重新打开讨论）**：与训练口径一致的实时 mbo 需 Databento Plus $1,750/月 + 流量 ~$29/月（实时 XNAS.ITCH mbo 仅 Plus/Unlimited 含；历史数据按量 $1.20/GB 不需订阅）——限价单模拟结果出来后，$1,780/月 是否值得重新进入测算区间，待真实 fill rate 落定后定夺
- **关键定稿**：
  - 不重叠采样（label 窗口 = 持仓窗口 = 6 分钟，样本不重叠）
  - 无止损时 R 口径 = 双向 risk = entry price（收益率是 R 的特例，两口径严格相等）
  - 训练/测试解耦（train.py 固化模型，test.py 载入，改止损不重训）
  - **坏点清洗（2026-08-29 增）**：分钟数据剔除 close≤0（盘后聚合伪影），否则做空假赚 100% 虚增收益
- 数据备份：GitHub Release `tick-2025-08-23`（2025 全量 + 2026 增量 + manifest）、`tick-full-2026-08-08` / `tick-full-2026-08-06`

## 用法

```bash
# 训练（固化模型，约 5 分钟）
python3 train.py --H 6 --train-days 8 --test-days 2            # 全特征
python3 train.py --H 6 --price-only                             # 纯价格特征

# 测试（载入模型，改止损无需重训）
python3 test.py --H 6 --stop-atr 2                              # 止损 2×ATR
python3 test.py --H 6 --stop-atr 0                              # 无止损（risk=entry 口径）
python3 test.py --H 6 --stop-atr 0 --price-only                 # 纯价格版
```

## 数据说明

- 训练数据：Databento XNAS.ITCH（QQQ mbo 全量订单簿）——2026-06~08（26 交易日）+ 2025-06~09（92 交易日），共 41GB 本地
- 实盘数据（若上线）：Databento Plus 订阅实时 mbo（$1,750/月 + 流量）——成交价口径（富途免费分钟 K）已被证伪不含此 edge
- `data_tick/` 与 `models/` 已 gitignore（数据/模型不进 git）

## 风险与局限（诚实）

- 仅 QQQ 单标的；时间上已跨 2025-2026 两个年份（118 天、4000+ 笔），但跨标的未验证
- **限价单 fill 是触及口径的乐观上界**（触及 ≠ 队列轮到你成交，价格瞬间穿过时被动单大概率没轮到）——真实 fill rate 是实盘化的决定性未知数，待 order_id 级委托队列模拟
- 净收益 +24~40 bps/笔，对滑点与成本假设敏感（限价口径 maker 0 + taker 3bps；市场动荡时点差扩大、撤单重挂的隐性成本未计）
- 实盘需 Plus 订阅 $1,780/月，edge 必须稳定覆盖月费后才有净价值——待真实 fill rate 落定后重算
- 信号密集时段一次一仓会跳过信号；跳过的信号分布是否随机未严格验证
