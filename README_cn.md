<p align="center">
  <img src="assets/logo.svg" width="320" alt="Intraday logo" />
</p>

<h1 align="center">Intraday —— Markowitz 的日内策略研究本体</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/Version-0.1.0-blue.svg" alt="Version: 0.1.0" />
  <img src="https://img.shields.io/badge/Focus-Intraday%20research-4F46E5.svg" alt="Focus: Intraday Research" />
  <img src="https://img.shields.io/badge/Markets-US%20%28QQQ%29-16C784.svg" alt="Markets: US (QQQ)" />
  <img src="https://img.shields.io/badge/Type-Research-FF1493.svg" alt="Type: Research" />
</p>

<p align="center">🌐 <a href="README.md">English</a></p>

**Intraday** 是量化策略 agent **Markowitz**（马科维茨）的**私有研究本体仓库**：日内尺度的策略研究——订单流（mbo）特征、分钟级机器学习信号、walk-forward 验证、执行口径模拟。本仓库由原 QuantStrategistAgent 项目拆分而来，体系共三个仓库、各司其职：

| 仓库 | 定位 | 可见性 |
|---|---|---|
| [QuantStrategistAgent](https://github.com/xhqing/QuantStrategistAgent) | 公开门面——Markowitz 的对外展示面 | 公开 |
| **Intraday（本仓库）** | 私有研究本体——日内策略研究现场 | 私有 |
| [Swing](https://github.com/xhqing/Swing) | 公开组件——已交付的日 K 趋势跟随策略工具集 | 公开 |

> 这是一个研究仓库，不是产品：`intraday/` 下是实验与验证脚本，结论持续订正、留痕记录。**不用于实盘、不构成投资建议。** Markowitz 不盯盘、不下单——本仓库是独立的研究项目，研究结论以同口径回测为准。

---

## 研究什么

日内（分钟级）价格 / 订单流信号是否存在可兑现的 edge。研究路径（脚本与状态详见 [`intraday/README.md`](intraday/README.md)）：

1. **数据**：Databento XNAS.ITCH 的 QQQ mbo 全量订单簿（委托口径）→ 聚合为分钟级订单流 + 量价特征。
2. **模型**：LightGBM 分钟方向分类——H=6 分钟持有窗口、不重叠采样、滚动 walk-forward，训练 / 测试解耦（模型固化、改止损不重训）。
3. **验证**：跨年独立复验（2025 / 2026 同口径零调参）、信号衰减曲线（edge 存活窗口）、执行口径对照（市价 vs 限价）。
4. **纪律**：结论（含订正与推翻）全部记入 CHANGELOG 与 `intraday/README.md`；坏点清洗、执行成本显式建模。

## 当前研究状态（摘要）

- **已收官（2026-09-13）：主线信号是 label 伪影，深度特征分支测毕亦关闭——「分钟尺度 + 非专业延迟」约束下的订单流方向研究终局。**完整发现链：order_id 级队列模拟测得真实 fill 只有 46% / 41%（2025/2026）、每信号期望 −1.7 / −2.5bps；延迟扫描曲线异常平坦（0~60 秒全 −5.3bps）触发根因深挖：信号分钟 close（最后一张新增委托价）与同分钟末笔成交价的偏差，24% 信号 >10bps（16% >20bps）——做多信号锚点均值低于市场 −28.8bps、做空 +31.0bps，模型学到的「方向」主体是委托价锚点向市场的机械回归，不是真实价格方向。
- **决定性检验**：同 15 个委托流特征、label 换成交价（F 价进 F 价出）重训——AUC 0.5078（2025）/ 0.5146（2026）≈ 随机，净 −5.9 / −5.7bps，126 个 walk-forward 窗口 0 个正。**现有 15 个基础特征对真实价格方向无预测力**；与富途成交价口径 5.5 年 −6bps（独立数据源）交叉验证。深度特征分支（同日测毕关闭）：17 个文献背书特征（CKS 盘口 OFI、多档深度不平衡、撤单/成交流、事件微结构）同框架检验——AUC 0.5005（2025）/ 0.4854（2026），46 窗口 0 个正。学术 OFI 预测力集中于秒级（HFT 领域），分钟尺度已稀释干净。
- **方法学遗产（本项目的持久价值）**：① 委托价口径与成交价口径是两个世界，用委托价做 label 会引入锚点回归伪影——任何 mbo/盘口数据研究的前置检验；② 队列优先级模拟方法（事件语义实测 + 重放平衡验证 + 同信号对照）可复用；③ 纪律框架（坏点清洗、不重叠采样、walk-forward、同口径对照）被反复验证有效。
- 历史数字（AUC 0.603、+24bps 上限、触及口径 +23.4 等）在 `intraday/STRATEGY.md` 保留为「伪影世界内部自洽」的记录，附订正说明。

## 仓库结构

```
Intraday/
├── CLAUDE.md                      # 项目入口——角色定位、铁律、研究纪律（AGENTS.md 为其软链接）
├── README.md / README_cn.md       # 英 / 中 README（本文件）
├── intraday/                      # 研究本体：train / test / 验证 / 模拟脚本 + 研究状态 README
│   ├── data_tick/                 # mbo 全量订单簿数据（约 27GB，gitignore，已云端备份）
│   └── models/                    # 固化模型（gitignore，train.py 可重生）
├── .claude/skills/quant/          # quant skill：K 线数据层（富途 / 长桥 + parquet 缓存）+ 量化开发规范
└── assets/                        # logo
```

## 数据源

| 数据源 | 内容 | 角色 |
|---|---|---|
| **Databento**（XNAS.ITCH mbo） | QQQ 全量订单簿：2025-06~09（92 交易日）+ 2026-06~08（26 交易日） | 日内订单流研究主力源（已备份 GitHub Release） |
| **富途 OpenD**（`futu-api`） | 约 5.5 年分钟 K（1m/5m/15m，2021-01-21 起）+ 约 20 年日 K | 成交价口径对照、A 股 ETF 日内 T+0 方向 |
| **长桥 CLI** | 十几年日 K（分页，每页 1000 根） | 日 K 交叉验证 |
| 富途资金流 / 盘口快照 / 换手率 | 仅当日、无历史 | 不进回测 |

完整调研（含付费渠道）见 `.claude/skills/quant/data-sources.md`。

## 前置条件

- Python 3 及 `pandas` / `numpy` / `lightgbm` / `scikit-learn` / `databento` / `pyarrow`
- **富途 OpenD** 本地网关在运行（K 线数据层）
- **Databento** API key（历史 mbo 按量付费 $1.20/GB，不需订阅）

## 风险声明

量化策略是研究工具，不是收益保证。回测表现不代表未来收益——过拟合、市场状态切换、前视偏差都可能虚高历史指标；本仓库的限价单成交率还是触及口径的乐观上界。本仓库产出的是研究代码与实验结论，供用户研究决策参考；**不构成投资建议，也不执行任何交易。** 用户对每笔交易决策及其造成的损失承担全部责任。作者与贡献者不对任何交易损失承担责任。

---

## 引用与署名

本项目以 MIT 许可证发布，额外请求使用者在**使用、二次分发或基于本项目构建衍生作品**时，注明作者并引用项目地址：

- **作者：** All Contributors
- **项目：** Intraday —— Markowitz 的日内策略研究本体（QuantStrategistAgent 体系）
- **体系门面：** https://github.com/xhqing/QuantStrategistAgent

若你引用代码或基于本仓库二次开发，请在文档 / README / 致谢中保留以上出处。

---

## 许可证

[MIT](LICENSE.md) © 2026 All Contributors。
