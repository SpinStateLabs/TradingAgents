# Outcome — definition of done for a Volatility-Trader run

The run is **satisfied** only if every criterion below is true. Each is binary.

1. **Briefs produced.** The output contains between 1 and 5 trade briefs, and each brief states a direction (long/short), an entry, a stop-loss, at least one take-profit target, and a position size.

2. **Sizing obeys the risk rule.** For every brief, the position size is consistent with risking at most 2% of current paper equity: `|entry − stop| × quantity` is approximately 2% of equity (within rounding). No brief violates the caps (≤3 concurrent positions, ≤6% total portfolio risk, no single position notional >25% of equity).

3. **Volatility evidence is numeric.** Every brief cites at least one concrete number (realized volatility, ATR% of price, range expansion, or volume surge vs trailing average) that justifies why this pair was selected over the rest of the universe.

4. **Risk is defined per trade.** Every brief has an explicit stop-loss and its implied reward:risk is ≥ 1.5 (target distance ≥ 1.5× stop distance).

5. **Paper portfolio updated correctly.** The output shows the simulated portfolio after this run: open positions marked to current price, any stop/target hits closed with realized P&L, and cash + total equity recomputed. No stated position breaches the caps in criterion 2.

6. **Actionable and clean.** The deliverable is a single markdown brief a trader could act on without asking a follow-up question — no placeholders, no "TODO", no contradictory numbers.
