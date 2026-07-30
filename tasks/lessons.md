# Lessons

Rules earned from actual mistakes and corrections in this codebase. Each one
exists because something went wrong or was nearly missed. Reviewed at the start
of a session.

---

## L1 — Never apply a risk-limiting check to a risk-reducing action

**Learned:** 2026-07-29, from `RiskEngine.evaluate`.

Every sizing candidate in the engine answered "how much *more* risk may this
take?" and was applied to every trade. On an exit they inverted: at
`max_position_weight` the headroom is zero, so the exit was rejected with "no
room to add risk". The position that most needed closing was the only one that
could not be.

**Why it matters:** it fails silently and in the flattering direction. The
broken engine reported +39.6% against the fixed engine's +3.7%, because a
position trapped since 2016 had become accidental buy-and-hold. It also drifted
to 39% of equity against its own 15% cap and stayed there for ten years. A
control that has stopped controlling still returns numbers.

**How to apply:** before any gate, ask whether this action *adds* or *removes*
risk, and gate only the former. Concretely — reject-if-no-headroom, minimum
confidence, maximum trades per day, gross-exposure ceilings, kill switches: all
of these are entry gates. An exit path must be reachable from every state the
system can reach, including the states it should never have reached.

The same bug appeared twice in one session. The second was in the new agent: a
`min_annual_vol` guard, correct for entries, was evaluated before the exit branch
and suppressed exits during quiet declines. **Assume this class is present
wherever a guard sits above a branch that handles both directions.**

---

## L2 — A bug fix that makes the backtest worse is still a bug fix

**Learned:** 2026-07-29, same defect.

Fixing L1 cut reported return from +39.59% to +3.72% and Sharpe from 0.70 to
0.34. The instinct is to distrust the fix. The fix was right; the old number was
an artifact of a frozen book.

**Why it matters:** the self-improvement loop is built to try variants and
promote the best. Run against a broken control it will reliably "discover" that
breaking the control improves returns, because in a 12-year bull market being
unable to sell *is* an edge. That is not overfitting to noise, it is overfitting
to a defect, and no amount of walk-forward or deflated-Sharpe correction catches
it — every fold agrees.

**How to apply:** when a change moves performance materially, A/B it against the
unmodified code with the same data and the same strategy, and explain the delta
mechanically before accepting either number. Here the explanation was one line:
the last fill was 2016-03-22. If the delta cannot be explained by a mechanism,
neither result is evidence yet.

---

## L3 — Resolve an ambiguous request before building, not after

**Learned:** 2026-07-29, from "create a basic fundamental trading strategy agent".

"Fundamental" reads as either fundamental analysis or a basic/baseline strategy.
The two need entirely different work: one needs a new point-in-time data feed, a
schema table and a cross-sectional engine; the other needs none of that and
runs on existing data. Building the wrong one wastes the whole session.

**How to apply:** when a request has two readings that imply materially
different builds, establish the facts that distinguish them first — here, that
there is no fundamentals table in `schema.sql` and no API key in `.env` — then
put the choice to Don with those facts attached. Facts first, then the question.
Asking without the facts just moves the guessing.

---

## L4 — Verify a scorecard against arithmetic you did yourself

**Learned:** 2026-07-29.

`scorecard.score` is well-written and well-tested, and it was still worth
re-deriving total return, CAGR, volatility, Sharpe and max drawdown straight
from the raw equity curve. They matched to 6 decimals. That match is what makes
the report evidence rather than an assertion.

**How to apply:** for any performance figure that will inform a capital
decision, compute it twice by different routes. Cheap, fast, and the only thing
that distinguishes a working metric from a plausible one. Also compute the
benchmark independently — do not let the same code produce both the strategy and
the thing it is measured against.

---

## L5 — Report the exposure-adjusted comparison alongside the raw one

**Learned:** 2026-07-29.

A long-only agent capped at `max_position_weight` (15% under MODERATE) holds
~85% cash by construction. Measuring it against a fully-invested index and
reporting −269% alpha answers a question nobody asked.

**Why it matters:** the honest framing is a decomposition, not a single number.
Here: 71.5% time in market × 7.03% median position ≈ 5.0% average exposure; 5.0%
of the index's +272.9% is ~+13.7% expected against +3.72% actual. Two thirds of
the gap is the exposure ceiling, one third is the strategy's timing being
value-destroying. Those are different problems with different fixes, and a
single alpha figure hides both.

**How to apply:** always report which constraint bound the size and how often
(`RiskDecision.binding_constraint` is already there — count it). Then state
whether the shortfall is exposure or skill. `kelly` bound 45 of 53 entries here,
which located the real problem in the edge estimate rather than in the trading
rule.

---

## L6 — Slice the trailing window; never hand a strategy the whole series

**Learned:** 2026-07-29, from the first version of `BaselineTrendAgent`.

`on_bar` called `cursor.history()` with no argument and recomputed rolling
statistics over the entire history every bar: O(n² × window). On 3020 bars the
lookahead verification exceeded a two-minute timeout. Switching to
`cursor.history(self.warmup_bars)` produced byte-identical results in 3.02s.

**Why it matters:** beyond speed, it is a tighter correctness guarantee. A
strategy that only ever requests its trailing window cannot use what it never
asked for, and that is auditable by inspection. There is now a test asserting
every `history()` call requests exactly `warmup_bars`.

**How to apply:** request the minimum window the statistics need, and assert it.
When optimising anything in the signal path, prove output equality against the
pre-optimisation run before accepting the speedup — the whole value of the change
is that it changed nothing.

---

## L7 — Test overlapping triggers as units, not through a price series

**Learned:** 2026-07-29, writing the exit tests.

Four exit triggers — trend break, volatility spike, hard stop, trailing stop —
overlap. Any drop deep enough to breach a hard stop usually breaks the trend
average too, and `trend_break` is checked first. A series-level test would pass
while three of the four triggers were dead code.

**How to apply:** where triggers are ordered and overlapping, drive the decision
function directly with synthesised inputs, one trigger at a time, plus explicit
precedence tests. Keep the series-level test as well, but as an integration
check, not as coverage of the individual branches.

---

## L8 — State known limitations with their direction of error

**Learned:** 2026-07-29.

Three limitations shipped with the baseline agent: it cannot see its own fills,
stops are evaluated on closes not intrabar lows, and prices exclude dividends.
Each was documented with which way it biases the result — missed trades never
phantom ones, later stops so pessimistic drawdowns, understated absolute CAGR
for both sides so the comparison is unaffected.

**Why it matters:** "known limitation" with no direction attached is unusable. A
reader cannot tell whether the reported number is a floor or a ceiling. With the
direction stated, a pessimistic result can be trusted as a lower bound and acted
on immediately.

**How to apply:** every caveat gets a direction of error and, where possible, a
magnitude. If the direction is unknown, say so explicitly — that is itself the
most important thing to report about it.

---

## L9 — An incomplete L1 fix is still an L1 bug

**Learned:** 2026-07-29, from the task 14 exit-reachability work.

The decision loop implemented "exits are always reachable" with a fresh
close-only `Mandate` for reductions, which correctly bypasses mandate expiry and
permission. An adversarial review then showed the same L1 trap survived one level
down: `min_confidence` and `max_trades_per_day` are checked in
`RiskEngine.evaluate` *before* the reduction branch, so once the day hit its
trade cap a stop-loss was still rejected — the exact "the position that most
needs closing is the one that cannot be" failure, reachable in the live loop.

**Why it matters:** L1 already warned this class recurs "wherever a guard sits
above a branch that handles both directions." The mandate-level fix looked
complete and even had passing tests, because those tests only exercised the
states the fix covered (`trades_today=0`, `confidence=1.0`). A partial fix with
green tests is more dangerous than no fix, because it reads as done.

**How to apply:** when fixing an exit-reachability trap, enumerate *every* gate
between entry and the reduction branch and confirm each is skipped for
reductions — not just the one that prompted the fix. Then write a test that puts
the system in each trapping state (at the trade cap, below the confidence floor,
under a contradicting bias) and asserts the exit still fires. The engine now
determines `reducing` up front and skips all three entry gates; the kill switch
is the sole deliberate exception.

---

## L10 — A `data/` ignore rule silently ate the whole data package

**Learned:** 2026-07-29, while committing task 13.

`.gitignore` carried a bare `data/` rule for the local data lake. Because it has
no leading slash, it matched **every** directory named `data` at any depth —
including the source package `spintrader/data/`. The entire data layer (store,
feeds, schema) had never actually been committed; a `feat(data)` commit existed
but was empty of those files, and `spintrader.cli` imported a package that was
not in git. It went unnoticed because everything runs off the on-disk files.

**Why it matters:** a fresh clone was broken and no test caught it — the files
were present locally, so the suite was green against a tree git did not have.

**How to apply:** anchor ignore rules for top-level artefact dirs with a leading
slash (`/data/`, not `data/`), and after adding one, run `git ls-files
<source>/` on any source dir that shares the name to confirm it is still tracked.
When a commit claims to add files, `git show HEAD:<path>` is the cheap check that
it actually did.
