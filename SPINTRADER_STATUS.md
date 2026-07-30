# SpinTrader — build status and handoff

Last updated: 2026-07-29. Branch `spintrader-foundation`, 11 commits ahead of
the `upstream` fork point (tasks 13 and 14 are uncommitted working-tree changes).
**725 tests passing on Windows** (21 skipped — `hmmlearn` has no Python 3.14
Windows wheel, so the HMM tests skip cleanly) and **746 on the GB10**, where the
HMM tests run. Tasks 13, 14 and 19 have been exercised end-to-end on the GB10
against the live Kraken endpoint and the real TimescaleDB.

Nothing is pushed to a remote. Nothing trades live.

---

## Where things run

| | |
|---|---|
| **Author here** | `D:\Apps\SpinTrader` (Windows). Full tooling, fast edit loop. |
| **Runs here** | `spinner@10.0.0.62:~/spintrader-src` (GB10 / DGX Spark). |
| **Sync** | one-way, `ops/sync.sh` or a `tar \| ssh` stream. **Never edit on the GB10** — the next sync discards it. |
| **SSH** | needs `-i ~/.ssh/gx10_ed25519`; plain `ssh` fails, it is not the default identity. |

The GB10 is the runtime: TimescaleDB 2.29 on `:5433`, Ollama on `:11434`,
IB Gateway on `127.0.0.1:4002`.

### Live services on the GB10

```bash
systemctl --user status spintrader-collector    # 1m bars, continuous
systemctl --user list-timers                    # daily refresh, 22:30 Toronto
docker ps                                       # spintrader-db, spintrader-ibgw, ollama
```

The collector matters: Kraken's REST OHLC endpoint reaches back only 720 bars,
so at 1-minute resolution **minute history can only be accumulated forward**.
Downtime is permanent data loss, not a delay.

---

## Verification scripts — run these first

```bash
cd ~/spintrader-src && export PYTHONPATH=.
.venv/bin/python ops/check_gb10_env.py    # db, ollama, data paths
.venv/bin/python ops/check_gb10.py        # both LLM tiers, live generation
.venv/bin/python ops/check_kraken.py      # 6 permissions, validate-only order
.venv/bin/python ops/check_ibkr.py        # gateway, paper account
.venv/bin/python -m spintrader.cli balances
```

All five passed as of the last session.

---

## What is built

| # | Component | Where |
|---|---|---|
| 1 | Repo scaffold on the TradingAgents fork | `spintrader/` alongside untouched `tradingagents/` |
| 2 | GB10 provisioning | `ops/provision.sh` |
| 3 | Tiered LLM router | `spintrader/llm/router.py` |
| 4 | Venue abstraction: Kraken, IBKR, paper | `spintrader/venues/` |
| 5 | TimescaleDB store + Kraken/yfinance feeds | `spintrader/data/` |
| 6 | HMM regime detection | `spintrader/quant/` |
| 7 | Risk engine, kill switch, mandate | `spintrader/risk/engine.py` |
| 8 | Portfolio ledger, multi-currency P&L | `spintrader/portfolio/ledger.py` |
| 9 | Walk-forward backtester + scorecard | `spintrader/backtest/` |
| 10 | 17 investor personas + voting panel | `spintrader/agents/` |
| 11 | Attribution, reliability, promotion gate | `spintrader/loop/` |
| 12 | Live 1m WebSocket collector | `spintrader/data/kraken_ws.py` |
| 13 | Deep 1m history from Kraken `/Trades` | `spintrader/data/kraken_trades.py` |
| 14 | Two-tier decision loop (fast quant + slow LLM mandate) | `spintrader/loop/` |
| 15 | Maker-first execution + maker/taker fee accounting | `spintrader/loop/decision_loop.py`, `spintrader/venues/paper.py` |
| 19 | Improvement cycle: candidate factory, research memory, orchestrator | `spintrader/research/`, `spintrader/loop/improvement.py` |

---

## What remains

Ordered by dependency.

- **16 — LLM recalibration agent** (daily). Guardrails already specified.
- **17 — News/sentiment ingestion.** Adapt `tradingagents/dataflows/reddit.py`
  and `stocktwits.py` rather than rewriting.
- **18 — Adaptive tier escalation.** `PanelVerdict.escalate` is already set; the
  LLM voter records it but the deep-model re-adjudication is not yet wired (there
  is a hook in `spintrader/loop/voting.py`).

---

## Tasks 13 & 14 — the running system

### 13 — deep 1m history (`spintrader/data/kraken_trades.py`)

Pages Kraken's public `/Trades` endpoint forward and aggregates trades into 1m
bars, reaching back years — where the OHLC endpoint stops at 720 bars and the WS
collector only goes forward. The one non-obvious rule: a minute is finalised
only once a trade in a *later* minute is seen, so a minute split across a
1000-trade page boundary yields **one** complete bar, not two partial ones (the
held bucket carries across pages). The forming final minute is never written and
is re-acquired on the next run. Resumable from the last stored bar.

```bash
python -m spintrader.cli data backfill-1m BTC-USD --years 2      # long, resumable
python -m spintrader.cli data backfill-1m --since 2023-01-01 --end 2023-02-01
```

Kraken rate-limits a deep backfill constantly (`EGeneral:Too many requests` after
~30 rapid pages), so `backfill_1m` backs off exponentially and retries the same
page rather than aborting the job. Verified on the GB10: backfilled BTC-USD from
2026-07-26 to the live edge — 188 pages, 5874 1m bars into TimescaleDB, riding
through rate limits, contiguous and resumable.

### 14 — two-tier decision loop (`spintrader/loop/`)

* `context.py` — the causal market snapshot both tiers share.
* `voting.py` — the persona-vote layer the panel needed but lacked. `LLMVoter`
  queries methodologies against the GB10 tiers (batched quick-then-deep, across
  assets); `BootstrapVoter` derives one quant opinion under a single lens so the
  panel correctly flags it low-conviction. Which one runs is a **runtime config**,
  not a fork.
* `mandate.py` — votes → panel verdict → `Mandate` (empty permits nothing;
  regime risk is the portfolio max; HOLD/abstention does not permit).
* `decision_loop.py` — the fast loop routes intents through the **same**
  `RiskEngine`/`PaperVenue`/`Ledger` the backtester uses (`LiveCursor` mirrors
  `ReplayCursor`); the slow loop refreshes the mandate.

```bash
python -m spintrader.cli loop mandate BTC-USD ETH-USD           # dry: print the mandate
python -m spintrader.cli loop run BTC-USD --max-ticks 5         # paper, bootstrap
python -m spintrader.cli loop run BTC-USD --llm --with-regime   # paper, LLM panel + HMM
```

Two decisions, made explicitly (confirmed with Don):

1. **Exits are always reachable.** A reduce/close intent is evaluated against a
   fresh close-only `Mandate`, so a held position can be exited even under an
   expired or unpermitted real mandate (lessons L1). New risk still answers to
   the real mandate. **The kill switch still halts everything, exits included**,
   pending a human reset — that boundary is deliberate, not an oversight.
2. **Slow loop = LLM panel with a quant bootstrap fallback**, injectable.

`loop run` is **paper-only by construction** — it forces `TradingMode.PAPER` and
never touches the three live switches. Arming live remains a human act.

An adversarial review of this changeset caught the exit-reachability decision
being *incomplete*: the close-only mandate bypassed mandate expiry/permission,
but `min_confidence` and `max_trades_per_day` were still checked before the
reduction branch in `RiskEngine.evaluate`, so a stop-loss was still trapped once
the day hit its trade cap. That is now fixed at the engine: a reduction is
identified up front and **skips every entry-only gate — confidence floor, daily
trade cap, and directional bias** — pinned by
`ExitReachabilityUnderEntryGatesTests`. The one remaining halt on an exit is the
**kill switch**, which is deliberate (a tripped switch means stop and wait for a
human) and is the sole exception.

---

## Task 19 — the improvement cycle (generative half)

The scoring half (walk-forward backtester, `PromotionGate`, `TrialLedger`,
reliability tracker) already refused almost everything. This is the generative
half that feeds it, in `spintrader/research/` and `spintrader/loop/improvement.py`:

* `research/factory.py` — the **agent factory**: enumerates configurations of
  strategy *families* (a strategy class + base config + grid), in a fixed order,
  filtering combinations a family would reject. Each config has a stable content
  hash over family + params. Two families ship, forecasting expected return in
  opposite ways: `BaselineTrendAgent` (buy strength) and `MeanReversionAgent`
  (buy weakness) -- so the search attacks the *edge*, not just the trend rule's
  parameters. The improvement cycle re-backtests a champion with its own family's
  class. New families drop in behind the same interface.
* `research/memory.py` — the **research memory**: records every candidate
  evaluated per objective (promoted or rejected, and why), so a round never
  re-tests — or re-counts — a configuration. Optionally persisted via the
  research cache so runs continue the search rather than restart it.
* `loop/improvement.py` — the **orchestrator**. Each fresh candidate is
  walk-forward backtested and run through the promotion gate against the current
  champion (re-backtested for an honest same-data comparison); the best
  challenger that clears the gate is promoted.

```bash
python -m spintrader.cli improve SPY --asset-class equity --interval 1d --csv data/SPY_1d.csv
python -m spintrader.cli improve BTC-USD                    # crypto 1m, from the store
```

**The load-bearing invariant:** every evaluated candidate is counted as a
lifetime trial *before* its result can promote anything — the gate increments
the `TrialLedger` on every `evaluate`, and the only skips are configs already
evaluated (already counted). Verified against real SPY data: four candidates,
four trials, deflated Sharpe falling 0.88 → 0.57 as the count rose, all
rejected. Nothing promoted is the correct, common outcome for a weak family —
the loop refusing to ship noise is the feature.

Also verified on the GB10 against **real 1-minute BTC** (4475 bars from the
backfill above): every candidate lost money out-of-sample and was rejected — the
expected result at minute cadence, where costs exceed the edge. This surfaced a
real bug, now fixed: `run_backtest` annualised the Sharpe by the cost model's
daily factor rather than the bar interval, which on 1m data inflated it ~38x and
would have bypassed the gate (lessons L11).

---

## Hard-won constraints — do not rediscover these

**Model tiers cannot be co-resident.** `qwen3:30b-a3b` (18 GB, 91 tok/s) and
`GLM-4.5-Air:Q6_K` (99 GB, 14.6 tok/s) total 117 GB of 121 GB. Ollama evicts one
to load the other, ~35 s per switch. `LLMRouter.run_batched()` enforces
all-quick-then-all-deep. Batch across assets too, not just within one.

**Pass `think: false` to qwen3 for schema-constrained calls.** Ollama routes a
reasoning model's output into a separate `thinking` field and applies the JSON
grammar there, leaving `response` empty. `complete_json` defaults to this.

**Minute holding is arithmetically impossible.** Measured on live Kraken data:
BTC 1m sigma is 0.0633%, a taker round trip costs 0.52% — an 8.2 sigma hurdle.
At a 1-hour hold it is 1.4 sigma and feasible. Decide every minute, hold for
hours.

**Kraken's OHLC endpoint serves only the most recent 720 bars.** `since` cannot
reach past that window. Hourly gives 30 days, daily ~2 years. Verified against
the live endpoint. Do not write a backwards-paging loop; there was one and it
was useless.

**Kraken asset codes are inconsistent.** Pre-2018 assets carry a class prefix
(`XXBT`, `ZUSD`, `ZCAD`); newer ones do not (`BNB`, `USDC`). A lookup keyed on
`CAD` silently finds nothing and reads as zero. `normalise_asset()` handles both
plus the `.S`/`.M`/`.F` staking suffixes. Kraken also returns **HTTP 200 with an
error array**, so failures look like successes to a naive client.

**Kraken permission probing:** `EOrder:*` errors mean the permission *works* —
the request authenticated and reached order validation. Only
`EGeneral:Permission denied` is a real permission failure.

**IBKR returns `-1` or `nan` for the touch outside RTH**, not an error. Unhandled
this produces a Quote with a negative spread. Use `SettledCash`, not
`TotalCashValue`, for available funds — the latter includes unsettled proceeds a
cash account cannot spend.

**PSR/DSR are defined on the per-period Sharpe.** Feeding them an annualised
value inflates the statistic by √252 for daily data and makes everything look
significant. The functions take `periods_per_year` and de-annualise internally.
The formula needs **full** kurtosis, not excess.

**Bar timestamps are CLOSE times, everywhere.** Kraken (REST and WS) and
yfinance all label by open or session date; all three are converted. Storing
open-time is the most common source of off-by-one-bar lookahead.

---

## Money and account reality

| | |
|---|---|
| Kraken | USD 701.52, BNB 0.5095 (~$291), dust. **API key has no withdrawal permission.** |
| IBKR | CAD 100, **cash account** — T+1 settlement, no shorting, no margin under $2,000 |
| Paper | `DUQ980038`, reset to 100 |
| Total | **~USD 1,063** |

`SPINTRADER_PAPER_EQUITY=100` overrides venue-reported equity in paper and
backtest, because IBKR seeds paper accounts with 1,000,000 CAD on 3.33× margin
and sizing against that validates strategies the real account cannot run.

**The IBKR account is CAD-based**, so the book is genuinely multi-currency. The
ledger handles it; a missing FX rate makes equity `complete=False` and
`to_risk_state()` raises rather than sizing against a partial view.

### Live trading is disarmed, on purpose

Three independent switches, all currently closed:

1. `SPINTRADER_LIVE_ENABLED=false`
2. `SPINTRADER_LIVE_VENUES=` (empty)
3. `READ_ONLY_API=yes` on the IB Gateway container

Per-order caps: Kraken $25, IBKR $10. Per-day: $150 / $50. These apply *after*
all strategy sizing.

**Claude does not arm live trading, place orders, or move funds.** The user
holds credentials and flips the switches. Credentials go in via
`ops/set_kraken_key.sh` / `ops/set_ibkr_creds.sh` (echo-disabled prompts) or
directly into `.env` — never through chat.

---

## Design commitments worth preserving

These are load-bearing. Several have tests written specifically to stop a future
refactor from "helpfully" relaxing them.

- **One submission path.** `Venue.submit()` is concrete on the base class;
  subclasses implement `_transmit`. No venue can be written that skips the gate.
- **Empty mandate allows nothing.** Defaulting open would let a failed mandate
  build authorise the whole universe.
- **Kill switch stays tripped** until a human resets it by name.
- **Mandates expire.** A stale hourly LLM view must not drive minute execution.
- **Abstention is not a neutral vote.** Abstainers leave the denominator; nine
  silent personas must not drag five informed ones toward inaction.
- **Reliability is driven by Brier skill, not P&L.** A persona does not choose
  its size, so it should not be judged on it.
- **Only advocates own a trade's P&L.** A dissenter who was overruled gets zero,
  not a debit.
- **Trial counts are cumulative and lifetime.** Resetting per round would let
  100 rounds of 10 candidates pass as 10 independent tests.
- **The backtester contains no simulation of trading.** It replays through the
  same `PaperVenue`, `RiskEngine` and `Ledger` that trade live.
- **`test_round_trip_at_a_flat_market_loses_money`** is load-bearing. If it ever
  passes at break-even the cost model is broken and every backtest downstream is
  optimistic.

---

## The demonstration to re-run if trust ever wavers

Forty coin-flip strategies on real BTC history produced Sharpe ratios from
−0.55 to **+1.05**. The best reads *"SIGNIFICANT"* when reported as a single
trial, and correctly reads **DSR 0.567, not significant** once `n_trials=40` is
accounted for.

That is the whole reason the promotion gate exists, and it is the check to run
before believing any result this system produces.
