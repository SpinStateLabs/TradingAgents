# Baseline trading agent + SPY paper backtest

Session: 2026-07-29. Scope agreed with Don up front: a **basic deterministic
baseline** agent (not fundamentals — see "Why not fundamentals" below), SPY
daily, MODERATE risk profile.

## Done

- [x] Mirror the package into a sandbox, establish a green baseline
      (288 passed, 21 skipped — `hmmlearn` absent)
- [x] Find and fix the exit-throttling defect in `RiskEngine.evaluate`
- [x] Write `spintrader/agents/personas/baseline_trend.py`
- [x] Write `spintrader/backtest/runner.py` + `spintrader backtest` CLI command
- [x] Write `scripts/fetch_bars_csv.py` so a backtest is reproducible off-LAN
- [x] Write 55 tests (`test_agents_baseline.py`, `test_risk_reductions.py`)
- [x] Run the SPY backtest (3020 bars, 2014-07-24 → 2026-07-28) + walk-forward
- [x] Verify: independent re-derivation, causality proof, A/B vs unpatched engine

Final: **343 passed, 21 skipped, 49 subtests** — no pre-existing test modified.

## Why not fundamentals

Don's phrasing was ambiguous. Established before writing code that a genuine
fundamental strategy is not currently buildable here, and got that decision
ratified:

1. `spintrader/data/` has only OHLCV feeds; `data/schema.sql` has no financials,
   earnings or estimates table. Nothing to read.
2. Fundamentals update quarterly, so 5 years × 1 ticker ≈ 20 independent
   observations. `BacktestEngine.run()` also takes a single `instrument`, so it
   cannot cross-sectionally rank a universe — which is where fundamental alpha
   lives.
3. No `ALPHA_VANTAGE_API_KEY` in `.env`. The free tier is 25 requests/day, and
   `OVERVIEW`-style ratios are a current snapshot, so a naive implementation is
   pure lookahead.

Deferred as its own piece of work: a point-in-time fundamentals feed
(`EARNINGS.reportedDate` is the only free field that is honestly
point-in-time), a `fundamentals` table, and a cross-sectional variant of the
engine that accepts a universe rather than one instrument.

## Defect found and fixed: reductions were sized as additions

`RiskEngine.evaluate` chose `target_weight` as the minimum of five candidates —
`kelly`, `vol_target`, `max_position_weight`, `gross_exposure_headroom`,
`position_headroom` — and applied them to *every* trade. All five answer "how
much more risk may this take?". Asked of a trade that removes risk they invert:

| Holding | Pre-fix outcome |
|---|---|
| 15% (at the `max_position_weight` cap) | **rejected**, `binding=position_headroom`, "no room to add risk" — 0% sellable |
| 12% | approved for **25%** of the position |
| 2% | approved in full |

So the position most in need of closing was the only one that could not be
closed, and a stop-loss could never fire at the cap.

### Measured impact (A/B, identical agent and data, only the engine differs)

| | return | maxDD | Sharpe | fills | rejected | last fill |
|---|---|---|---|---|---|---|
| unpatched engine | +39.59% | -7.70% | 0.70 | 9 | **96** (all `position_headroom`) | **2016-03-22** |
| patched engine | +3.72% | -2.38% | 0.34 | 105 | 0 | 2026-04-27 |

Read that carefully — **the bug made the numbers look better.** It bought in
2014, got trapped, and executed nothing for the next ten years and one month.
The stuck position drifted to **39.0% of equity against a 15% cap — a 2.6x
breach of the engine's own position limit, held unmanaged for a decade.** The
apparent +39.6% is scaled buy-and-hold produced by a broken control, not a
strategy.

The engine had lost control of the book. That is the finding, not the return.

### The fix (`spintrader/risk/engine.py`)

Three changes, all inside `evaluate`:

1. Classify the trade: `reducing = (SELL and held > 0) or (BUY and held < 0)`.
2. A reducing trade skips the additive candidates and is sized off the position
   it closes. Confidence floor, mandate, trade-count limit, venue minimums and
   the live notional gate all still apply.
3. `qty = abs(held_qty)` for a reduction, rather than round-tripping
   weight → notional → price. The weight is measured at the mid and the fill is
   at the touch, and that half-spread of slack left a dust position behind on
   every exit. A closing BUY is also capped at flat so it cannot flip into a long.

Pinned by `tests_spintrader/test_risk_reductions.py` (22 tests). Every one fails
against the unpatched engine. `EntrySizingUnchangedTests` proves entry sizing was
not loosened.

### Same bug class, second instance — in the new agent

The agent's `min_annual_vol` guard (which exists to stop Kelly demanding an
unbounded position when the volatility denominator collapses) was evaluated
*before* the exit branch. An orderly linear decline has near-zero realised
volatility, so a slow bleed — exactly the shape that needs an exit — suppressed
the exit entirely and trapped the position. Caught by a test, not by reading the
code. Exits are now evaluated first, before any gate that could suppress them.

## Results: `baseline_trend_v1` on SPY, MODERATE, 2014-07-24 → 2026-07-28

3020 daily bars, $1000 start, 252 periods/yr, T+1 cash account, long-only.

| | agent | SPY buy & hold |
|---|---|---|
| total return | +3.72% | +272.95% |
| CAGR | +0.31% | ~11.6% |
| annualised vol | 0.90% | 17.51% |
| Sharpe | 0.34 (95% CI −0.22 … 0.91) | 0.72 |
| **max drawdown** | **−2.38%** | **−34.10%** |
| fills | 105 | 1 |
| hit rate | 32.7% | — |
| profit factor | 1.81 | — |
| cost drag | 1.09% of gross P&L | — |
| rejections | **0** | — |

Walk-forward, 4 out-of-sample folds (755 train / 504 test / 102 embargo):
+0.22%, +2.91%, −0.12%, +1.27%. Stitched Sharpe 0.57 (95% CI −0.12 … 1.26),
maxDD −1.7%, **cost drag 83.2%**. Not significant.

### Verdict: it works, it is safe, and it is not good

Three separate claims, and they need separating:

1. **The wiring is correct.** Zero rejections over 105 orders, so the agent's
   intended-exposure tracking never desynced. The engine's scorecard matches an
   independent re-derivation from the raw equity curve to 6 decimal places.
   Truncating the series never changed an earlier decision, at four different
   cut points — no lookahead.
2. **Risk control is real.** −2.38% maximum drawdown against the index's
   −34.10%: a 14x reduction, and it sat out both the COVID crash and the 2022
   bear market. Against Don's stated objective (minimise drawdown and margin
   calls) this is the part that works.
3. **The returns are not there.** Sharpe 0.34 against buy-and-hold's 0.72, so
   it is worse *risk-adjusted*, not merely smaller. DSR 0.879 < 0.95, so it is
   not statistically distinguishable from noise even before any parameter
   search. The stitched walk-forward cost drag of 83.2% says the edge is barely
   larger than its own trading costs.

As a baseline that is a useful result: this is the floor, the floor is low, and
it is now measured rather than assumed.

### Why the return is so small — decomposed, not guessed

- time in market: **71.5%** of bars
- median sized position: **7.03%** of equity (range 0.80% … 15.00%)
- `kelly` was the binding constraint on **45 of 53** entries;
  `max_position_weight` on the other 8
- average exposure: 71.5% × 7.03% ≈ **5.0%**
- 5.0% of the index's +272.9% ≈ **+13.7%** expected; actual **+3.72%**

So roughly two thirds of the shortfall is the exposure ceiling and one third is
the strategy's own timing losing money against simply holding. Kelly binds
because the edge estimate is weak: median 0.98% against a variance of
0.16² = 2.56%, giving 0.0098/0.0256 × 0.20 × 0.709 ≈ 5.4%.

**The binding constraint is the edge estimate, not the trading rule.** Tuning
the moving-average windows would move nothing; a better forecast of expected
return would move everything. That is where a fundamentals or LLM agent has to
earn its place.

## Not done, deliberately

- **`min_confidence` and `max_trades_per_day` still gate exits.** These are the
  same bug class as the sizing defect — a risk-limiting check applied to a
  risk-reducing action. Under MODERATE, an exit needs confidence ≥ 0.62, and the
  6-trades-per-day cap can block a stop on a busy day. The persona sets exit
  confidence to 1.0 so it is unaffected today, but a future agent that reports
  honest uncertainty on an exit would be blocked from closing. Left alone
  because changing it alters documented behaviour and is Don's call, not mine.
- **Extending the `Strategy` protocol to pass `PortfolioState` to `on_bar`.**
  The persona cannot see whether its orders filled, so it tracks intent instead.
  Benign here (zero rejections) and one-directional when it does bite — it
  causes missed trades, never phantom ones — but it is a real gap.
- **Dividends.** Yahoo unadjusted prices exclude them, understating both the
  agent and the benchmark by ~1.3%/yr for SPY. The comparison is unaffected;
  the absolute CAGR figures are floors, not totals.
- **Intrabar stops.** Both stops are evaluated on closes. Daily bars cannot
  resolve whether the low preceded or followed the close, and assuming the
  favourable order is how backtests invent returns. Reported drawdowns are
  therefore pessimistic.
- **No parameter tuning at all.** `--trials` exists and feeds
  `scorecard.score(n_trials=...)`. Every future sweep must set it honestly or
  the deflated Sharpe becomes decoration.

## Files

New:
- `spintrader/agents/personas/baseline_trend.py`
- `spintrader/agents/personas/__init__.py`
- `spintrader/backtest/runner.py`
- `scripts/fetch_bars_csv.py`
- `tests_spintrader/test_agents_baseline.py` (33 tests)
- `tests_spintrader/test_risk_reductions.py` (22 tests)
- `data/SPY_1d.csv`, `artifacts/*` (equity curve, scorecard, report)

Modified:
- `spintrader/risk/engine.py` — reduction sizing (the defect above)
- `spintrader/cli.py` — `backtest` subcommand

## Reproduce

```bash
python scripts/fetch_bars_csv.py SPY --years 12 --out data/SPY_1d.csv
python -m spintrader.cli backtest SPY --csv data/SPY_1d.csv \
    --asset-class equity --aggression moderate --cash 1000 --out artifacts
```

On GB10, drop `--csv` to read from TimescaleDB instead. Exit code 2 means the
result is not significant after the multiple-testing adjustment.

## Next, in the order I would do it

1. **Decide the `min_confidence`/`max_trades_per_day`-on-exits question.** It is
   a live trap in the same family as the one just fixed.
2. **Widen the `Strategy` protocol** to pass portfolio state into `on_bar`.
   Removes the intent-tracking workaround entirely.
3. **Work on the edge estimate, not the rules.** The decomposition above says
   that is the only lever that matters. Anything that improves the forecast of
   expected return — fundamentals, regime conditioning via the existing
   `quant/regime.py`, an LLM analyst — should be measured as a replacement for
   `_edge()` and scored against `baseline_trend_v1` with `--trials` set truthfully.
4. **Then, and only then, a cross-sectional engine.** `max_position_weight` of
   15% against `max_positions` of 8 says the risk profile was designed for a
   diversified book. Single-instrument backtests will always look
   under-exposed because they are.
