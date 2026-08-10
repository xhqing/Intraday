# quant-swing — 日 K 趋势跟随策略工具

> 量化策略（Markowitz 开发）的用户 CLI 工具。**日 K 波段趋势跟随**，经 20 年 × 37 只美股历史验证（四重泛化测试全过）。策略完整文字版见 [STRATEGY.md](STRATEGY.md)。

## 快速开始

```bash
# 1. 更新数据（需富途 OpenD 在线；首次跑一次）
cd .claude/skills/quant && python3 data.py fetch-all --ktype DAY

# 2. 每天美股收盘后，检测全部标的信号
cd quant-swing && python3 signal.py

# 有 ✅ 触发信号 → 次日开盘买入（仓位/止损/trailing 见 STRATEGY.md）
```

## CLI 一览

| 命令 | 功能 |
|---|---|
| `python3 signal.py` | **检测信号**（每天收盘后跑）。`--symbols NVDA --verbose` 单标的详情；`--scheme 2` 换方案 |
| `python3 backtest.py` | **回测**。`--scheme 1-4` 四种方案；`--all` 对比全部；`--symbols --mode` 自定义 |
| `python3 analyze.py <维度>` | **数据分析**。维度：`monthly` 月收益 / `hold` 持仓 / `yearly` 按年 / `symbols` 标的排名 / `worst-month` 最差月 / `gap` 跳空 / `signal-day` 信号次日行为 |

## 四种方案

| 方案 | 标的 | 模式 | 特点 |
|---|---|---|---|
| **1（推荐）** | 37 美股 | 只做多 | **g 最高 +0.444%、波动最低**、最差月最浅 |
| 2 | 37 美股 | 多空 | 月胜率 65% 但波动翻倍（36.5%）、最差月 −52% |
| 3 | 37+6 低相关 ETF | 只做多 | 削危机月尾（2020-02 从 −3.6% → +13%）但 g 略降 |
| 4 | 37+6 低相关 ETF | 多空 | 夏普最高 1.29 但尾部最差（最差月 −47.7%） |

**推荐方案 1**：`python3 backtest.py --scheme 1` 查看完整回测。

## 目录结构

```
quant-swing/
├── README.md        # 本文件（项目说明）
├── STRATEGY.md      # 策略完整文字版（信号/执行/止损/trailing/仓位/历史表现/已证伪方案/局限）
├── common.py        # 共享：标的池/参数/数据加载（三个 CLI 共用的唯一接缝）
├── signal.py        # 检测信号 CLI
├── backtest.py      # 回测 CLI
└── analyze.py       # 数据分析 CLI
```

**解耦**：三个 CLI 互不依赖，只依赖 `common.py`；数据与回测引擎复用 quant skill（同一 data_cache、同一引擎，保证口径一致）。

## 数据依赖

- **数据源**：富途 OpenD（本机 `~/FutuOpenD/FutuOpenD.app`，需登录）。数据缓存在本机 `quant/data_cache/`（gitignore，可重生）。
- **备份仓库**：[xhqing/market-data-backup](https://github.com/xhqing/market-data-backup)（私有）。所有行情数据快照按 tag 挂 GitHub Release，防止免费数据源失效丢失：
  - `snapshot-YYYY-MM-DD`：富途日 K/分钟 K 快照（parquet 打包）
  - `tick-YYYY-MM-DD`：Databento 逐笔/订单簿数据（如 QQQ mbo）
  - 恢复：`quant/data.py restore <tag>`（日 K 快照）；tick 数据从 Release 手动下载（含 manifest.json SHA256 校验）
- **历史范围**：日 K 约 20 年（2006-2026，4930-5039 根/标的）。
- **成本假设**：美股 3bps/边（佣金+滑点）。

## 风险提示（必读）

1. **约 1/3 年份（震荡/危机）亏钱**——2011/2012 欧债、2022 熊市都是负年。要能承受连续 1-2 年震荡回撤。
2. **月胜率 54%**——接近一半月份浮亏，心理门槛高。
3. **隔夜跳空**——最差单笔 −5.58R，**实盘仓位 f 用 1%~1.7%（绝不用 2%）**。
4. **历史验证 ≠ 未来保证**——20 年数据验证的 edge 未来可能衰减，但零衰减的样本外验证说明当前可靠。
5. 本工具为量化策略信号与回测分析，**不构成投资建议**；决策与执行由使用者自行负责。

*策略逻辑与回测由 Markowitz（QuantStrategistAgent）开发。*
