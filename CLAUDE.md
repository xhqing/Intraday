# Intraday

> **Markowitz（马科维茨）日内策略研究本体**的私有仓库。原 QuantStrategistAgent 项目拆分为三仓库：[QuantStrategistAgent](https://github.com/xhqing/QuantStrategistAgent)（公开门面）+ **Intraday（本仓库，私有研究本体）** + [Swing](https://github.com/xhqing/Swing)（公开组件，日 K 策略工具集）。本仓库专注日内尺度的策略研究。

## 本仓库做什么

Markowitz 在这里做**离线、确定性**的日内研究：订单流（mbo）特征、分钟级 ML 信号、walk-forward 验证、执行口径模拟。**不盯盘、不下单、不发实时交易信号**——那是日内交易员 Victor（[DayTradingAgent](https://github.com/xhqing/DayTradingAgent)）的职责；本仓库的研究结论经回测验证后，作为 Victor 众多输入中的一个加权投票（体系上经门面仓库衔接）。

一句话定位：**本仓库是 Markowitz「日内 edge 是否存在、能否兑现」的研究现场——一切结论以同口径回测为准，订正留痕。**

## 三条铁律（最高优先级，用户立）

1. **时间必带市场时区**：港股用北京 / 港时（`+08:00`），美股用美东（`EDT -04:00` / `EST -05:00`）。一律 ISO 8601 带偏移。禁止无时区时间戳。
2. **输出全量化**：每个字段要么是**数值**（价格 / 量 / 比率 / 百分比），要么是**有限枚举**（离散值）。**禁止定性文字**（如「5 日均线上行」「动能减弱」）。这是量化区别于主观交易的本质。
3. **schema 与回测协同设计**：只有输入（K 线 / 订单流数据）和输出（schema）都可量化，回测才能**精准复现**（同数据 + 同参数 → 同结果）。

> 判定依据不许写文字，必须转成一组数值 `features`（如均线值、斜率、量比、ATR、订单流不平衡）。定性结论（方向 / 状态）只作为这些数值的离散枚举结果输出。

## 研究纪律

1. **同口径复现**：同一数据 + 同参数 + 同代码 → 同结果。实验脚本无随机、无全局状态、无未来函数；训练 / 测试解耦（`train.py` 固化模型，`test.py` 载入复验，改止损不重训）。
2. **只用有历史、可回测的数据**：日 K / 分钟 K（富途、长桥）+ Databento mbo 订单流（有历史）。富途资金流 / 盘口快照无历史，不进回测。
3. **结论必须有数**：胜率 / AUC / bps / 样本数 / 窗口数，禁止「感觉有效」；被推翻的结论保留订正痕迹（CHANGELOG + `intraday/README.md`）。
4. **优化目标 = 期望收益（avg_R / bps）**，不是胜率；执行口径（市价 / 限价、成本、fill rate）必须显式建模——「回测有 edge」与「可交易」是两个问题。
5. **坏点清洗**：分钟数据剔 close≤0 等伪影，否则统计结论虚高（曾因 856 个坏分钟虚增 43bps，见 CHANGELOG 2026-08-29 条目）。

## 目录与产物

- `intraday/`：研究本体（脚本 + 研究状态 README，当前状态以它为准）。
- `.claude/skills/quant/`：quant skill——数据层（K 线拉取 + parquet 缓存）与量化开发规范；`intraday/` 的脚本直接复用其 `data_cache/`。输入输出契约与回测设计见其 `schema.md`。
- `AGENTS.md → CLAUDE.md`、`.pi/skills → .claude/skills` 为软链接（多工具共用同一份指令 / skill）。
- 产物 = 可复现实验代码 + 结论记录（`intraday/README.md` + CHANGELOG）；不产实时信号。

## 数据约束（实测）

- **Databento XNAS.ITCH mbo**：QQQ 全量订单簿（委托口径），2025-06~09（92 交易日）+ 2026-06~08（26 交易日）共 118 天，本地约 27GB（`intraday/data_tick/`，已 gitignore，备份在 market-data-backup Release）。历史按量 $1.20/GB 不需订阅；实时 mbo 需 Plus 订阅 $1,750/月。
- **富途 OpenD 分钟 K**：约 5.5 年（2021-01-21 起），1m / 5m / 15m 均有，单次 `max_count` 上限 10 万根，分页拉全量。**成交价口径对照的主力源。** 符号 `市场.代码` 前缀（`HK.02800`）；离线时 `open ~/FutuOpenD/FutuOpenD.app` 启动。
- **长桥日 K**：单次上限 1000 根，分页拼接（每页约 4 年）可拉十几年；港股美股均支持。符号 `代码.市场` 后缀（`02800.HK` / `SPY.US`）。
- **富途资金流 / 盘口 / 换手率**：无历史，不进回测。

数据源调研（含付费第三方渠道、踩坑备忘）见 `.claude/skills/quant/data-sources.md`。

## 工作规则

通用工作规范三件套（verify-before-report / file-operation-priority / tmp-dir-for-artifacts）已全文并入全局 `~/.claude/CLAUDE.md`『工作规则』节、随全局 CLAUDE.md 自动加载，项目不再维护副本（权威源见 CapabilityManagerAgent 仓库的 `claude/CLAUDE.md`）。

> 事实性 / 数值性结论先验证再陈述（实体归属、算术、日历、API 字段、费率），禁止把推测当事实——这条是量化工作的底线：研究代码里的每一个数值都必须有据可查、可复现。
