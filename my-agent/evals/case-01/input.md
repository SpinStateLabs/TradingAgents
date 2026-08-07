# Eval case 01 — v0 kickoff input

This is the first live run's input (also the deployment's recurring task). It is the
regression baseline: after the winning v0 run, its verified output is saved as
`expected.md` in this folder and re-run against future agent versions.

Universe: BTC_USDT, ETH_USDT, SOL_USDT, BNB_USDT, XRP_USDT, DOGE_USDT, ADA_USDT,
AVAX_USDT, LINK_USDT, TON_USDT, TRX_USDT, DOT_USDT, POL_USDT, LTC_USDT, BCH_USDT,
NEAR_USDT, APT_USDT, ARB_USDT, OP_USDT, SUI_USDT

Task: run the standard volatility scan + paper-trading cycle (see `first_prompt.txt`),
grade against `outcome.md`.

Note: crypto volatility data is fresh every run, so the "right answer" ages out by design.
The rubric (`outcome.md`) is the real eval; this case is the regression baseline for
"does a new agent version still produce a correct, well-sized, well-reasoned brief."
