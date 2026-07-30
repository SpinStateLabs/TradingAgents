"""Baseline trend-following persona -- the floor every other agent must beat.

Why this exists
---------------
Before any LLM-driven or fundamentals-driven agent can be called good, something
has to define what "good" means. This persona is that reference point: a
deterministic, fully specified, long-only trend follower with no free
parameters fitted to the test set and no model to overfit. If a sophisticated
agent cannot beat it out-of-sample after costs, the sophistication is
decoration.

It is deliberately dull. Dullness is the feature -- a baseline that is itself
tuned is not a baseline, it is another candidate strategy wearing a baseline's
name, and comparing against it flatters everything.

Design constraints this persona respects
----------------------------------------
**It never sizes.** :class:`~spintrader.risk.engine.RiskEngine` owns sizing.
The persona reports three numbers -- ``edge``, ``confidence``, ``volatility`` --
and the risk engine decides how much of the book that justifies. A strategy that
also sized itself would have two places to encode risk appetite and they would
disagree.

**It never reads the future.** Every statistic is trailing, computed only from
:meth:`ReplayCursor.history`, which physically cannot return bars beyond the
current one. The causality of the rolling helpers is separately enforced by
:func:`spintrader.quant.features.assert_causal`.

**It is stateless across folds.** :meth:`fit` is a documented no-op. There is
nothing to fit, so walk-forward folds cannot leak state through it.

A known limitation, stated rather than hidden
---------------------------------------------
The ``Strategy`` protocol gives ``on_bar`` no sight of the ledger, so this
persona cannot see whether its orders actually filled. It therefore tracks its
*intended* exposure and emits only on transitions. If the risk engine rejects an
entry, the persona believes it is long when it is flat and will sit out until the
next regime flip.

That failure mode is one-directional: it causes missed trades, never phantom
ones, so it understates performance and can never inflate it. It is also
observable -- a backtest whose rejection histogram shows no rejected entries had
no desync at all. Check ``BacktestResult.rejections`` before trusting a run.
The real fix is to widen the protocol so ``on_bar`` receives the portfolio
state; that is a change to the engine, not to this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np

from spintrader.core.types import Bar, Instrument, Side, to_decimal
from spintrader.quant.features import log_returns, periods_per_year, rolling_std
from spintrader.risk.engine import Mandate, TradeIntent

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class TrendReading:
    """Everything the persona computed on one bar.

    Exposed so a run can be audited after the fact. A signal that cannot be
    reconstructed from stored numbers cannot be debugged when it loses money.
    """
    close: float
    fast: float
    slow: float
    trend_strength: float       # fast/slow - 1
    annual_vol: float
    above_slow: bool
    vol_ok: bool
    bullish: bool
    edge: float
    confidence: float


class BaselineTrendAgent:
    """Long-only dual-moving-average trend follower with volatility gating.

    Entry requires *all* of:

    * close above the slow moving average -- direction
    * fast moving average above the slow -- trend confirmation
    * trailing annualised volatility at or below ``vol_ceiling`` -- refuses to
      buy into a crisis, which is where trend followers take their worst losses

    Exit fires on *any* of:

    * close below the slow moving average -- the trend is over
    * volatility above ``vol_ceiling`` -- the regime changed underneath us
    * close at or below ``stop_pct`` under the entry mark -- hard stop
    * close at or below ``trail_pct`` under the highest close since entry --
      trailing stop, which is the component that actually limits drawdown

    Both stops are evaluated on closes, not intrabar lows. Bar data cannot
    resolve whether a low preceded or followed a close, and assuming the
    favourable order is the single most common way a backtest invents returns
    it will never see live. This makes the persona's stops *later* than a real
    intrabar stop, so reported drawdowns are pessimistic rather than optimistic.
    """

    name = "baseline_trend_v1"

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 100,
        vol_window: int = 20,
        vol_ceiling: Decimal | str | float = "0.40",
        stop_pct: Decimal | str | float = "0.05",
        trail_pct: Decimal | str | float = "0.08",
        edge_floor: Decimal | str | float = "0.005",
        edge_ceiling: Decimal | str | float = "0.06",
        interval: str = "1d",
        continuous: bool = False,
        min_annual_vol: Decimal | str | float = "0.01",
        spread_bps: Decimal | str | float = "5",
        name: str | None = None,
    ) -> None:
        if fast_window < 2:
            raise ValueError("fast_window must be at least 2")
        if slow_window <= fast_window:
            raise ValueError(
                f"slow_window ({slow_window}) must exceed fast_window "
                f"({fast_window}); otherwise the crossover has no meaning"
            )
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")

        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_window = vol_window
        self.vol_ceiling = to_decimal(vol_ceiling)
        self.stop_pct = to_decimal(stop_pct)
        self.trail_pct = to_decimal(trail_pct)
        self.edge_floor = to_decimal(edge_floor)
        self.edge_ceiling = to_decimal(edge_ceiling)
        self.interval = interval
        self.continuous = continuous
        self.min_annual_vol = to_decimal(min_annual_vol)
        # Must match the spread the engine hands the venue. If the quote used
        # for sizing is tighter than the quote used for filling, every position
        # is sized off a price the venue will not honour and the error compounds
        # silently across the run.
        self.spread_bps = to_decimal(spread_bps)
        if name:
            self.name = name

        # Annualisation is resolved once, at construction, so an unknown
        # interval fails loudly here rather than silently mis-scaling every
        # volatility for the whole run.
        self._ann_scale = math.sqrt(
            periods_per_year(interval, continuous=continuous)
        )

        # Intended exposure. See the module docstring on why this is intent
        # rather than fact.
        self._long = False
        self._entry_mark: float | None = None
        self._peak_mark: float | None = None
        self._last_reading: TrendReading | None = None

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """Bars required before any signal is defensible."""
        return max(self.slow_window, self.vol_window) + 2

    @property
    def is_long(self) -> bool:
        """The persona's *intended* exposure, not a confirmed position."""
        return self._long

    @property
    def last_reading(self) -> TrendReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "vol_window": self.vol_window,
            "vol_ceiling": str(self.vol_ceiling),
            "stop_pct": str(self.stop_pct),
            "trail_pct": str(self.trail_pct),
            "edge_floor": str(self.edge_floor),
            "edge_ceiling": str(self.edge_ceiling),
            "interval": self.interval,
            "continuous": self.continuous,
            "warmup_bars": self.warmup_bars,
        }

    # -- strategy protocol -------------------------------------------------

    def fit(self, bars: Sequence[Bar]) -> None:
        """No-op: there is no model here.

        Kept explicit rather than omitted. A baseline with a silent fit step is
        a baseline that might be leaking the training window, and the reader
        should not have to check.
        """
        self.reset()

    def reset(self) -> None:
        self._long = False
        self._entry_mark = None
        self._peak_mark = None
        self._last_reading = None

    def on_bar(
        self, cursor, instrument: Instrument, mandate: Mandate,
    ) -> Sequence[TradeIntent]:
        # Request only the trailing window the statistics need. Asking for the
        # whole series would make every bar O(len(history)) and the run
        # O(n^2 * window) -- on twelve years of daily bars that is the
        # difference between one second and several minutes. It is also a
        # tighter causality guarantee: the persona cannot use what it never
        # asked for.
        history = cursor.history(self.warmup_bars)
        if len(history) < self.warmup_bars:
            return ()

        reading = self._read(history)
        self._last_reading = reading

        # Exits are evaluated first, and before any gate that could suppress
        # them. A guard that blocks entering is prudent; the same guard blocking
        # an *exit* traps the position it was meant to protect. That is the same
        # error the risk engine made by sizing reductions from its remaining risk
        # budget, and it is worth being explicit that this ordering is the fix
        # rather than an accident of layout.
        if self._long:
            reason = self._exit_reason(reading)
            if reason is None:
                return ()
            self._long = False
            self._entry_mark = None
            self._peak_mark = None
            # Floor the reported volatility. A reduction is sized from the held
            # quantity rather than from volatility, but the intent must still
            # carry a usable number: a near-zero one would demand an unbounded
            # position from anything that did divide by it.
            return (self._exit_intent(
                instrument, cursor,
                max(to_decimal(reading.annual_vol), self.min_annual_vol),
                reason,
            ),)

        if reading.annual_vol <= float(self.min_annual_vol):
            # Vol-target and Kelly both divide by volatility. A near-zero
            # denominator produces an unbounded position, so refuse to *open*
            # one rather than emit a number the risk engine will scale into the
            # whole book.
            return ()

        volatility = to_decimal(reading.annual_vol)

        if not reading.bullish:
            return ()
        if not mandate.allows(instrument.key):
            return ()
        if mandate.bias_for(instrument.key) < ZERO:
            # The mandate says short; a long-only persona has nothing to say.
            return ()

        self._long = True
        self._entry_mark = reading.close
        self._peak_mark = reading.close
        return (self._entry_intent(instrument, cursor, reading, volatility),)

    # -- signal ------------------------------------------------------------

    def _read(self, history: Sequence[Bar]) -> TrendReading:
        closes = np.array([float(b.close) for b in history], dtype=float)
        close = float(closes[-1])

        fast = float(closes[-self.fast_window:].mean())
        slow = float(closes[-self.slow_window:].mean())
        trend_strength = (fast / slow - 1.0) if slow > 0 else 0.0

        # Trailing realised volatility. rolling_std is causal by construction;
        # only the final element is consumed, so this is the value observable
        # at this bar and no earlier.
        returns = log_returns(closes)
        annual_vol = float(
            rolling_std(returns, self.vol_window)[-1] * self._ann_scale
        )

        above_slow = close > slow
        vol_ok = annual_vol <= float(self.vol_ceiling)
        bullish = above_slow and fast > slow and vol_ok

        edge = self._edge(trend_strength)
        confidence = self._confidence(trend_strength, annual_vol)

        # Track the running peak while long, for the trailing stop.
        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return TrendReading(
            close=close, fast=fast, slow=slow,
            trend_strength=trend_strength, annual_vol=annual_vol,
            above_slow=above_slow, vol_ok=vol_ok, bullish=bullish,
            edge=edge, confidence=confidence,
        )

    def _edge(self, trend_strength: float) -> float:
        """Expected forward return, clipped.

        Trend strength is a weak proxy for expected return and an unclipped one
        is worse than useless: Kelly divides it by variance, so a single
        extreme reading would demand the entire book. The clip is the honest
        admission that this signal's magnitude carries much less information
        than its sign.
        """
        floor = float(self.edge_floor)
        ceiling = float(self.edge_ceiling)
        return min(ceiling, max(floor, trend_strength))

    def _confidence(self, trend_strength: float, annual_vol: float) -> float:
        """Conviction in [0, 1], blending trend clarity and calm.

        Anchored at 0.60 so a marginal signal sits *below* the MODERATE
        profile's 0.62 floor and is filtered by the risk engine rather than by
        a threshold duplicated here. Two components, weighted toward trend:
        volatility already enters sizing through vol-targeting, so letting it
        dominate confidence would penalise it twice.
        """
        ceiling = float(self.edge_ceiling)
        strength_score = min(1.0, max(0.0, trend_strength / ceiling))

        vol_span = float(self.vol_ceiling)
        vol_score = min(1.0, max(0.0, (vol_span - annual_vol) / vol_span))

        blended = 0.65 * strength_score + 0.35 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended))

    def _exit_reason(self, reading: TrendReading) -> str | None:
        if not reading.above_slow:
            return "trend_break"
        if not reading.vol_ok:
            return "vol_spike"
        if self._entry_mark is not None and self._entry_mark > 0:
            floor = self._entry_mark * (1.0 - float(self.stop_pct))
            if reading.close <= floor:
                return "hard_stop"
        if self._peak_mark is not None and self._peak_mark > 0:
            floor = self._peak_mark * (1.0 - float(self.trail_pct))
            if reading.close <= floor:
                return "trailing_stop"
        return None

    # -- intents -----------------------------------------------------------

    def _entry_intent(
        self, instrument: Instrument, cursor, reading: TrendReading,
        volatility: Decimal,
    ) -> TradeIntent:
        return TradeIntent(
            instrument=instrument,
            side=Side.BUY,
            edge=to_decimal(reading.edge),
            confidence=to_decimal(reading.confidence),
            volatility=volatility,
            quote=cursor.quote(self.spread_bps),
            strategy=self.name,
        )

    def _exit_intent(
        self, instrument: Instrument, cursor, volatility: Decimal, reason: str,
    ) -> TradeIntent:
        # An exit is a rule firing, not a forecast, so it carries full
        # conviction -- otherwise the risk engine's min_confidence floor could
        # veto a stop-loss, which is exactly backwards.
        #
        # The edge is set high enough that a full exit survives even if the
        # engine's reduction path is bypassed and Kelly sizes the trade. It
        # costs nothing here (reductions are capped at the held quantity) and
        # it means a stop cannot be silently throttled into a partial sale.
        defensive_edge = max(self.edge_ceiling, volatility * volatility * 4)
        return TradeIntent(
            instrument=instrument,
            side=Side.SELL,
            edge=defensive_edge,
            confidence=ONE,
            volatility=volatility,
            quote=cursor.quote(self.spread_bps),
            strategy=f"{self.name}:{reason}",
        )


__all__ = ["BaselineTrendAgent", "TrendReading"]
