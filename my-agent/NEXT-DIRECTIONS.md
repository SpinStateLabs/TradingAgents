# Next directions — Volatility-Trader

A numbered sequence of planned releases. v0 (live now once launched) is the crypto
volatility scanner with a **simulated paper account** in a memory store, run on a daily
schedule. Everything below is deferred deliberately, each with the exact mechanism.

Before promoting any new agent version to the deployment: **re-run `evals/` and diff the
verdicts** against the current baseline.

## v1 — Real paper-trading exchange fills
- **What:** Replace the memory-ledger fills with real simulated fills from a crypto
  exchange paper/testnet API (Alpaca Crypto paper, or Binance/Bybit testnet).
- **Why later:** needs an exchange API credential that isn't in hand right now.
- **How:** create a Vault `environment_variable` credential (`EXCHANGE_API_KEY` /
  `_SECRET`) + set the environment `networking: limited` with the exchange host allowlisted;
  give the agent a thin order helper that posts to the testnet. The portfolio state stays
  the source of truth; fills become real (slippage, partial fills) instead of modeled.

## v2 — Live, ask-before-execute
- **What:** Real (small) live account; the agent drafts each order and waits for your
  explicit approval before it sends.
- **Why later:** real money — wire only after paper shows a consistent edge.
- **How:** real exchange account behind the same API; put the order tool behind an
  `always_ask` permission policy, and add an interface (Console confirmation, or a small
  generated UI) that surfaces the confirmation so you can approve/deny each trade.

## v3 — Live, fully automated
- **What:** The agent places real orders unattended, no human in the loop.
- **Why later:** highest risk; only once evals prove consistent positive paper/live-gated
  alpha over enough runs.
- **How:** remove the `always_ask` gate once trusted; keep every hard risk cap
  (2%/trade, 6% portfolio, 25% single-position, RR ≥ 1.5) in the system prompt.

## v-next — Equity options volatility plays
- **What:** IV rank screens, straddles/strangles, vol-crush around earnings.
- **Why later:** scheduled for a later version; richest volatility surface but new data +
  execution stack.
- **How:** add an options data source and an options-capable broker (Tradier / IBKR);
  new rubric criteria for IV/greeks.

## v-next — Event-driven trigger
- **What:** In addition to (or instead of) the daily scan, fire a run the moment a
  volatility signal trips (a large move, an IV/VIX spike).
- **How:** your backend calls `POST /v1/deployments/<id>/run` from a webhook handler when
  the signal fires.

## Backlog
- More volatility signals (funding-rate skew, orderbook imbalance, options IV term
  structure once options land).
- **Dreams** consolidation over the growing `closed_trades` log to distill what setups
  actually worked (research preview).
- A generated results dashboard over the Files/Sessions API (equity curve, win rate,
  per-setup P&L).
