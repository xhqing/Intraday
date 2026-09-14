<p align="center">
  <img src="assets/logo.svg" width="320" alt="Intraday logo" />
</p>

<h1 align="center">Intraday — Markowitz's Private Intraday Research Base</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/Version-0.1.0-blue.svg" alt="Version: 0.1.0" />
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

> This is a research repository, not a product: `intraday/` holds experiment and validation scripts, with conclusions continuously revised and recorded. **Not for live trading, not financial advice.** Markowitz does not watch the market or place orders — this is an independent research project, and conclusions stand on their own walk-forward backtests.

---

## What Is Researched

Whether intraday (minute-scale) price / order-flow signals carry a realizable edge. The research path (scripts and status in [`intraday/README.md`](intraday/README.md)):

1. **Data**: Databento XNAS.ITCH full QQQ mbo order book (order-book basis) → aggregated into minute-level order-flow + price/volume features.
2. **Model**: LightGBM minute-direction classification — H=6 minute holding window, non-overlapping sampling, rolling walk-forward, decoupled train/test (frozen models; changing stops needs no retrain).
3. **Validation**: independent cross-year replication (2025 / 2026, identical protocol, zero re-tuning), signal-decay curves (edge survival window), execution-basis comparison (market vs limit orders).
4. **Discipline**: every conclusion (including corrections and reversals) is recorded in the CHANGELOG and `intraday/README.md`; bad-tick cleaning and explicit execution-cost modeling.

## Current Research Status (Summary)

- **Concluded (2026-09-13): the main-line signal was a label artifact, and the deep-feature branch was opened and closed the same day — order-flow direction research is finished under the "minute scale + non-professional latency" constraint.** The full discovery chain: the order_id-level queue simulation showed a real fill rate of only 46% / 41% (2025/2026) with a per-signal expectation of −1.7 / −2.5 bps; the latency sweep curve came out suspiciously flat (−5.3 bps across 0–60 s), which triggered a root-cause dig: the signal-minute close (last submitted-order price) deviates from the last actual trade price by >10 bps in 24% of signals (>20 bps in 16%) — long-signal anchors average −28.8 bps below the market, short-signal anchors +31.0 bps above. The learned "direction" was mostly the mechanical reversion of the order-price anchor toward the market, not real price direction.
- **Decisive test**: retrained with the same 15 order-flow features but a trade-price label (fill-price entry/exit) — AUC 0.5078 (2025) / 0.5146 (2026), i.e. random; net −5.9 / −5.7 bps; 0 of 126 walk-forward windows positive. **The existing 15 basic features have no predictive power for real price direction**; cross-validated by the independent Futu trade-price basis (−6 bps/trade over 5.5 years). Deep-feature branch (opened and closed the same day): 17 literature-backed features (CKS book-state OFI, multi-level depth imbalance, cancel/trade flow, event microstructure) tested under the same protocol — AUC 0.5005 (2025) / 0.4854 (2026), 0 of 46 windows positive. Academic OFI predictive power lives at the second scale (HFT territory); minute aggregation dilutes it to zero.
- **Methodological legacy (the durable value of this project)**: ① order-price basis vs trade-price basis are two worlds — using order prices as labels injects an anchor-reversion artifact (a mandatory pre-check for any mbo/order-book research); ② the queue-priority simulation method (event-semantics verification + book-replay balance check + same-signal comparison) is reusable; ③ the discipline framework (bad-tick cleaning, non-overlapping sampling, walk-forward, same-basis comparison) proved effective.
- Historical numbers (AUC 0.603, +24 bps upper bound, +23.4 bps touch basis, etc.) remain recorded in `intraday/STRATEGY.md` as an internally consistent artifact world, with correction notes attached.

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
