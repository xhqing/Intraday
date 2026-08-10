<p align="center">
  <img src="assets/logo.svg" width="320" alt="Markowitz logo" />
</p>

<h1 align="center">Markowitz — Quant Strategy Agent (HK / US Equities)</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/focus-quant%20strategies-4F46E5.svg" alt="Focus: Quant Strategies" />
  <img src="https://img.shields.io/badge/markets-HK%20%2F%20US-16C784.svg" alt="Markets: HK / US" />
  <img src="https://img.shields.io/github/last-commit/xhqing/QuantStrategistAgent" alt="Last Commit" />
  <img src="https://img.shields.io/badge/Type-AI%20Agent-FF1493.svg" alt="Type: AI Agent" />
</p>

<p align="center">🌐 <a href="README_cn.md">简体中文</a></p>

**Markowitz** is a personified AI agent dedicated to **quantitative strategy development** for Hong Kong and US equities, built on [Claude Code](https://claude.com/claude-code). It designs backtestable trading strategies as deterministic code, runs them against historical data to calibrate their credibility, and hands the result to **Victor** — the day-trading agent ([DayTradingAgent](https://github.com/xhqing/DayTradingAgent)) — as one weighted, quantified vote among Victor's many inputs.

> This is **not** a traditional software project. There is no application to `npm install`. The repository *is* the agent: its behavior is shaped by the `skills` under `.claude/` (plus the user's global rules), which Claude Code loads as Markowitz's operating discipline.

---

## Who is Markowitz?

The agent is personified as **Markowitz** — a quantitative strategy developer. The name honors **Harry Markowitz**, father of Modern Portfolio Theory and Nobel laureate, whose mean-variance framework laid the foundation of quantitative investing.

Markowitz does **not** watch the market, place orders, or emit real-time trading signals — that is Victor's job. Markowitz's work is **offline** and **deterministic**: turn trading ideas into pure-function code, backtest them on years of historical K-lines, and let the backtest — not intuition — assign each strategy its credibility.

The edge comes from mathematical rigor that can be reproduced:

- **Fully quantified output.** Every field is a number or a finite enum — no qualitative prose ("momentum weakening"). This is what separates quant from discretionary.
- **Backtest-validated credibility.** A strategy does not grade itself; its confidence comes from the historical win-rate / avg_R of its signal category.
- **Pure functions, no lookahead.** Same bars + same params → same output, always. No future data leaks into a decision.

---

## Relationship to Victor (DayTradingAgent)

Markowitz and Victor are **separate, complementary projects**:

| | Markowitz (this repo) | Victor ([DayTradingAgent](https://github.com/xhqing/DayTradingAgent)) |
|---|---|---|
| **Job** | Design & backtest quant strategies | Watch the market & emit trading signals |
| **Mode** | Offline, deterministic code | Real-time, LLM judgment |
| **Output** | Strategy code + credibility table | 🟢/🔴/🟡/🟠 signals for human execution |
| **Touches account?** | No — never | No — signal mode only |

Markowitz's product is **one weighted input** to Victor — a "quantified voter" with historically validated credibility. It does not replace Victor's judgment; the discretionary factors (capital flow, order book, news, macro) stay with Victor.

> Why split? An LLM cannot be backtested (it is persona-like, not 100% reproducible, subject to lookahead bias like a human). But deterministic strategy code *can* be. So Markowitz formalizes the backtestable subset of trading logic and lets the backtest grade it.

---

## Core Philosophy

| Principle | What it means |
|---|---|
| **Facts first** | Verify entity identity, arithmetic, calendar, API fields, fees before asserting. Never pass a guess off as a fact. |
| **Fully quantified** | Every output field is a number or finite enum; the judgment basis goes in numeric `features`, never prose. |
| **Backtest-graded confidence** | Strategies only classify (emit a `category`); win-rate / avg_R / samples come from the backtest, not self-rating. |
| **Pure & reproducible** | Same cached data + same params + same code = identical backtest. No randomness, no global state, no lookahead. |
| **avg_R optimization** | Optimize average risk-multiple P&L (expectancy), not win-rate or total return. |

---

## How Markowitz Works

<img src="assets/markowitz_workflow.svg" width="100%" alt="Markowitz workflow: Design → Backtest → Calibrate → Deliver" />

1. **Design** — write a strategy in `strategies/` as a pure function `evaluate(bars, params) -> dict | None` that returns the strategy / tactic schema **without** confidence.
2. **Backtest** — `backtest.py` walks the K-lines bar-by-bar (no future data), records every trade's R-multiple, and groups results by `category`.
3. **Calibrate** — aggregate per category into `confidence_table.json` (`win_rate` / `avg_R` / `samples`); small-sample categories are discounted.
4. **Deliver** — the strategy + its credibility table become a weighted input for Victor.

The mechanics of one backtested trade (real data — daily trend-following on `US.TSLA`, 2014):

<img src="assets/backtest_example.png" width="100%" alt="A real backtest trade: signal at t close → t+1 open entry → trailing stop exit → R-multiple" />

- Signal fires at the close of bar `t` → fill at the **open of `t+1`** (conservative — never at the signal bar's own price).
- Exits are mechanical: fixed stop / take-profit / timeout for intraday, trailing stop (wins-only) for daily trend-following — whichever triggers first, filled at the trigger bar's open.
- `R = (exit − entry) / risk`, where `risk = stop_atr × ATR` — every trade's result is a risk-multiple, directly comparable across strategies.
- Costs (slippage + commission + stamp duty) are subtracted per trade **before** any performance number is reported.

---

## Backtested Track Record

The first delivered strategy — **daily trend-following** (long-only, 37 US large-caps & ETFs, trailing stop at 3×ATR) — is backtested on ~20 years of daily K-lines (2006–2026, 2,086 trades):

<img src="assets/performance.png" width="100%" alt="Backtested performance: cumulative net R, per-year net avg_R, per-stock net avg_R, R distribution" />

| Metric | Value |
|---|---|
| Period / trades | 2006–2026 · 2,086 trades |
| Net avg_R (after 6 bps round-trip costs) | **+0.25 R** per trade |
| Growth rate `g` at 2% risk per trade | **+0.44% per trade** |
| Win rate | 40.8% (low win-rate, high payoff — the design) |
| Positive-`g` stocks | 34 / 37 |
| Positive years | 18 / 21 (negatives: 2008, 2014, 2022 — crisis / sideways regimes) |

Validation beyond the headline numbers (cross-sectional, yearly, parameter robustness, strict walk-forward with zero decay) is documented in [`quant-swing/STRATEGY.md`](quant-swing/STRATEGY.md). Honest notes: intraday (minute-K) strategies were **systematically falsified** — momentum and mean-reversion both show no edge net of HK stamp duty (a documented conclusion, not a gap); the daily scale is where the edge survives. Backtested performance does not guarantee future results — see the [Risk Disclaimer](#risk-disclaimer).

---

## Data Sources

| Source | Depth | Role |
|---|---|---|
| **Futu OpenD** (`futu-api`) | ~5.5 yr minute K (1m/5m/15m, since 2021-01-21); ~20 yr day K | **Primary for intraday backtests** |
| **Longbridge CLI** | 10+ yr day K (paginated, 1000/page) | Day-K backtest + cross-validation |
| Capital flow / order book / turnover | Current only, **no history** | Excluded from strategies (stays with Victor) |

Full survey (incl. paid third-party channels) lives in `.claude/skills/quant/data-sources.md`.

---

## Repository Structure

```
QuantStrategistAgent/
├── CLAUDE.md                      # Entry point — Markowitz's role, ironclad rules, workflow
├── LICENSE.md                     # MIT
├── README.md                      # This file (English)
├── README_cn.md                   # Chinese README
│
├── .claude/
│   ├── settings.local.json        # Local-only: permissions + autoMemoryDirectory (gitignored)
│   ├── settings.local.example.json # Template for settings.local.json (tracked)
│   ├── memory/                    # AutoMemory store (project-level, tracked — not gitignored)
│   │
│   └── skills/
│       ├── quant/                 # Quant strategy spec — the heart of Markowitz
│       │   ├── SKILL.md           # Master file: execution spec + ironclad rules
│       │   ├── README.md          # Directory purpose + design principles + conventions
│       │   ├── schema.md          # Strategy I/O contract + backtest design
│       │   ├── data-sources.md    # Intraday data-source survey
│       │   ├── strategies/        # Each strategy one pure-function file
│       │   ├── data.py            # K-line fetcher (Futu minute + Longbridge day)
│       │   ├── backtest.py        # Backtest driver → category credibility table
│       │   └── data_cache/        # parquet snapshots (gitignored — regenerable)
│       ├── anysearch/             # Core search skill (ships with the agent)
│       └── find-skill/            # Core skill-discovery skill (ships with the agent)
```

---

## Prerequisites

To actually run Markowitz, you need — outside this repo:

- [Claude Code](https://claude.com/claude-code)
- **Futu OpenD** local gateway running (minute-K backtest primary source)
- **Longbridge CLI** configured (day-K backtest + cross-validation)
- Python 3 with `pandas` (and `futu-api` / longbridge bindings as needed)
- Optionally copy `.claude/settings.local.example.json` → `.claude/settings.local.json` and set `autoMemoryDirectory` to your machine's absolute path to store AutoMemory inside the project

Without these, the repo still reads as a complete spec of *how a disciplined quant-swing agent should behave*.

---

## Current Stage

Markowitz has moved past MVP. The schema, backtest engine, and data layer are complete, and the **first validated strategy is delivered** — daily trend-following on 37 US stocks (see [Backtested Track Record](#backtested-track-record)). It ships as a standalone toolset in [`quant-swing/`](quant-swing/README.md) with three CLIs: `check_signal.py` (daily signal scan), `backtest.py` (schemes / custom backtests), and `analyze.py` (performance analysis).

What was tried and set aside, honestly:

- **Intraday (minute-K) strategies were systematically falsified** — no predictive edge net of HK stamp duty; documented in `.claude/skills/quant/SKILL.md` to prevent re-treading the dead end.
- The **day-K trend-following edge survived** strict validation (34/37 stocks positive, 18/21 years positive, parameter-robust plateau, walk-forward zero decay — details in `quant-swing/STRATEGY.md`).
- Next milestones: broaden the pool (low-correlation additions), track live signal checks, and re-validate as new history accrues.

---

## Risk Disclaimer

Quantitative strategies are research tools, not guarantees. Backtested performance does not predict future results — overfitting, regime change, and lookahead bias can all inflate historical metrics. Markowitz produces strategy code and credibility estimates for the user's research judgment; **it is not financial advice and executes no trades.** The authors and contributors assume no liability for trading losses.

---

## Attribution

This project is released under the MIT License, and you are additionally asked to **credit the author and cite the source** whenever you use, redistribute, or build upon it:

- **Author:** All Contributors
- **Project:** Markowitz — Quant Strategy Agent (HK / US Equities)
- **Source:** https://github.com/xhqing/QuantStrategistAgent

If you fork, reference, or derive from this repository, please retain this attribution — the author name, the project name, and the repository URL — in your documentation, README, or acknowledgements.

---

## License

[MIT](LICENSE.md) © 2026 All Contributors.
