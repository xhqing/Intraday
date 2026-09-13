<p align="center">
  <img src="assets/logo.svg" width="320" alt="Intraday logo" />
</p>

<h1 align="center">Intraday — Markowitz's Private Intraday Research Base</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/Focus-Intraday%20research-4F46E5.svg" alt="Focus: Intraday Research" />
  <img src="https://img.shields.io/badge/Markets-US%20%28QQQ%29-16C784.svg" alt="Markets: US (QQQ)" />
  <img src="https://img.shields.io/badge/Type-Research-FF1493.svg" alt="Type: Research" />
</p>

<p align="center">🌐 <a href="README_cn.md">简体中文</a></p>

**Intraday** is the **private research repository** of **Markowitz**, the quant-strategy agent: intraday-scale strategy research — order-flow (mbo) features, minute-level ML signals, walk-forward validation, and execution-modeling simulations. This repository was split out of the original QuantStrategistAgent project; the system now consists of three repositories, each with its own role:

| Repository | Role | Visibility |
|---|---|---|
| [QuantStrategistAgent](https://github.com/xhqing/QuantStrategistAgent) | Public facade — Markowitz's public face | Public |
| **Intraday (this repo)** | Private research base — where intraday research happens | Private |
| [Swing](https://github.com/xhqing/Swing) | Public component — the delivered day-K trend-following toolset | Public |

> This is a research repository, not a product: `intraday/` holds experiment and validation scripts, with conclusions continuously revised and recorded. **Not for live trading, not financial advice.** Markowitz does not watch the market or place orders — validated research conclusions become one weighted vote among the many inputs of Victor, the day-trading agent ([DayTradingAgent](https://github.com/xhqing/DayTradingAgent)).

---

## What Is Researched

Whether intraday (minute-scale) price / order-flow signals carry a realizable edge. The research path (scripts and status in [`intraday/README.md`](intraday/README.md)):

1. **Data**: Databento XNAS.ITCH full QQQ mbo order book (order-book basis) → aggregated into minute-level order-flow + price/volume features.
2. **Model**: LightGBM minute-direction classification — H=6 minute holding window, non-overlapping sampling, rolling walk-forward, decoupled train/test (frozen models; changing stops needs no retrain).
3. **Validation**: independent cross-year replication (2025 / 2026, identical protocol, zero re-tuning), signal-decay curves (edge survival window), execution-basis comparison (market vs limit orders).
4. **Discipline**: every conclusion (including corrections and reversals) is recorded in the CHANGELOG and `intraday/README.md`; bad-tick cleaning and explicit execution-cost modeling.

## Current Research Status (Summary)

- **Directional predictive power holds across years**: order-book-basis 6-minute direction — 2025 AUC 0.603 (net +24.0 bps/trade), 2026 AUC 0.589 (net +39.2 bps/trade).
- **Execution basis decides everything**: t+1 open market-order entry −4.8 bps (negative every month — the edge fears latency); a t+1 minute limit order at the signal price (δ=0) fills 92.5% on the touch basis with +23.4 bps expected per signal (the edge does not fear passive waiting).
- **The decisive unknown**: real fill rate (price touching ≠ your turn in the queue) — pending an order_id-level queue simulation; live trading needs a Databento Plus subscription ($1,780/month), to be re-evaluated once the real fill rate is known.
- **Falsified**: the trade-price basis (free Futu minute K) shows no edge over 5.5 years of retrained walk-forward (−6 bps/trade) — this edge is an order-book-basis product and does not exist in trade-price data.

## Repository Structure

```
Intraday/
├── CLAUDE.md                      # Entry point — role, ironclad rules, research discipline (AGENTS.md symlinks to it)
├── README.md / README_cn.md       # English / Chinese README (this file)
├── intraday/                      # Research core: train / test / validation / simulation scripts + status README
│   ├── data_tick/                 # Full mbo order-book data (~27GB, gitignored, backed up to cloud)
│   └── models/                    # Frozen models (gitignored, regenerable via train.py)
├── .claude/skills/quant/          # quant skill: K-line data layer (Futu / Longbridge + parquet cache) + quant dev spec
└── assets/                        # logo
```

## Data Sources

| Source | Content | Role |
|---|---|---|
| **Databento** (XNAS.ITCH mbo) | Full QQQ order book: 2025-06~09 (92 trading days) + 2026-06~08 (26 trading days) | Primary source for intraday order-flow research (backed up to a GitHub Release) |
| **Futu OpenD** (`futu-api`) | ~5.5 yr minute K (1m/5m/15m, since 2021-01-21); ~20 yr day K | Trade-price-basis comparison; A-share ETF intraday T+0 direction |
| **Longbridge CLI** | 10+ yr day K (paginated, 1000/page) | Day-K cross-validation |
| Futu capital flow / order-book snapshot / turnover | Current day only, no history | Excluded from backtests |

Full survey (incl. paid channels) lives in `.claude/skills/quant/data-sources.md`.

## Prerequisites

- Python 3 with `pandas` / `numpy` / `lightgbm` / `scikit-learn` / `databento` / `pyarrow`
- **Futu OpenD** local gateway running (K-line data layer)
- **Databento** API key (historical mbo is usage-based at $1.20/GB, no subscription needed)

## Risk Disclaimer

Quantitative strategies are research tools, not guarantees. Backtested performance does not predict future results — overfitting, regime change, and lookahead bias can all inflate historical metrics; the limit-order fill rate in this repo is still an optimistic touch-basis upper bound. This repository produces research code and experimental conclusions for the user's research judgment; **it is not financial advice and executes no trades.** The user assumes full responsibility for any trading decision and any resulting loss. The authors and contributors assume no liability for trading losses.

---

## Attribution

This project is released under the MIT License, and you are additionally asked to **credit the author and cite the source** whenever you use, redistribute, or build upon it:

- **Author:** All Contributors
- **Project:** Intraday — Markowitz's private intraday research base (QuantStrategistAgent system)
- **System facade:** https://github.com/xhqing/QuantStrategistAgent

If you reference code or derive from this repository, please retain this attribution in your documentation, README, or acknowledgements.

---

## License

[MIT](LICENSE.md) © 2026 All Contributors.
