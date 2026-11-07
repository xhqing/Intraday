# Intraday 项目规则（Worker 追加）

你在 **Intraday**（Mark 的日内策略研究仓库）里干活。本文件由 pi 在会话启动时自动追加到系统提示，只写本仓库特有的硬约束；Agent 身份与通用规则由 `AGENTS.md` 软链接提供，此处不重述。

## 你的角色

- 若你是 Mark 派来的 **Worker**：只做当前 worktree 里这一件事，干完汇报（改了什么 / 验证结果 / PR 号 / 遗留问题），不碰 local main、不建删 worktree。
- 这是**研究仓库**：不下单、不接实盘、不发实时信号；一切结论以同口径回测为准。

## 数据在哪、什么不能碰

- **tick / 订单簿数据（`intraday/data_tick/`，约 25GB）不在本机**：本体在 macmini 的 `~/Data/Intraday/data_tick`，另有 GitHub Release `market-data-backup` 备份。要跑依赖它的脚本，先说明并改用 `macmini_run` 在 macmini 上跑（或在 macmini 建好软链跑），**不要**往本机硬拉 25GB、不要删远端数据。
- **可重生运行时产物，一律不入库**：`intraday/data_cache/`（K 线 parquet）、`intraday/output/`（回测记录）、`intraday/models/`（LightGBM 固化模型）。需要时重新生成。
- **Databento API key 只从 `~/.zshrc` 读**（`fetch_mbo.py` 已如此实现）：key 不得写进任何被跟踪文件、日志或命令行参数。
- 富途 OpenD 在线才有 K 线（`127.0.0.1:11111`）；离线先执行 `open ~/FutuOpenD/FutuOpenD.app`。

## 口径铁律（踩过的坑，别再踩）

1. **label / 评估价必须用成交价 F**：用委托价 `close`（该分钟最后一张新增委托价）做标签会制造「锚点回归伪影」——2026-09 曾据此虚报 +24bps，最终被 `verify_flabel.py` 推翻（AUC≈0.51）。任何 mbo 研究先按 F 价口径检验。
2. **坏点清洗**：剔除 `close <= 0` 的分钟（盘后聚合伪影）；不清洗会做出「做空假赚 100%」。
3. **采样与验证**：不重叠采样（label 窗口 = 持仓窗口）、滚动 walk-forward、训练 / 测试解耦（`train.py` 固化模型 → `test.py` 只读模型，改止损不重训）。
4. **成本与成交率**：默认双边 6bps 已计入；限价成交率必须用 `queue_sim.py` 的 order_id 级队列口径，`limit_sim.py` 的触及口径只是**乐观上界**，不能当结论。
5. **跨年复验**：2025（92 天）与 2026（26 天）用同口径零调参各跑一遍，单年正不算成立。
6. **时区**：美股一律 `America/New_York`（EDT −04:00 / EST −05:00），ISO 8601 带偏移，不写裸时间戳。

## 结论与文档纪律

- 每条结论（含订正、推翻）都要写进根 `CHANGELOG.md` + `intraday/README.md`（研究状态）+ `intraday/CONCLUSIONS.md`（结题层）；**旧结论不删**，标「订正」并指向新结论。
- 已证伪的方向（分钟尺度订单流基础 / 深度特征、开盘动量、A 股隔夜反向）不重复试验；要重开必须先说明「这次为什么可能不同」。

## 常用入口

`python3 train.py --H 6` → `python3 test.py --H 6 --stop-atr 2`；跨年 `verify_2025.py --year 2025|2026`；延迟扫描 `latency_curve.py`；真实 fill `queue_sim.py`。脚本都在 `intraday/` 下运行。

## 更多

研究细节见 `intraday/CONCLUSIONS.md` 与 `intraday/STRATEGY.md`；数据源调研见主仓库 quant skill 的 `data-sources.md`；通用工作规则见全局 `~/.pi/agent/AGENTS.md`。
