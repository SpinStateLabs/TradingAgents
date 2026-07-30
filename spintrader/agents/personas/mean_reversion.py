"""Mean-reversion persona -- a genuinely different edge from the trend baseline.

Why this exists
---------------
The baseline session's decomposition was blunt: "the binding constraint is the
edge estimate, not the trading rule." Tuning the trend follower's windows moves
nothing; a *different forecast of expected return* moves everything. This is that
different forecast. Where :class:`~spintrader.agents.personas.baseline_trend.BaselineTrendAgent`
buys strength and expects it to continue, this buys weakness and expects it to
revert -- the opposite sign on the same price series. At minute cadence on crypto,
short-horizon reversion is where the honest edge tends to be, if there is one.

It is long-only, for the same reason the baseline is: the account is a cash
account that cannot short. It fades dips, never rallies.

Design constraints it respects (identical to the baseline, on purpose)
----------------------------------------------------------------------
* **It never sizes.** It reports ``edge``, ``confidence`` and ``volatility``;
  the risk engine decides how much. The expected-return estimate is the gap back
  to the mean, which is the honest thing a reversion process is forecasting.
* **It never reads the future.** Every statistic is trailing, from
  :meth:`ReplayCursor.history`, over exactly ``warmup_bars``.
* **Exits are evaluated first**, before any guard that could suppress them
  (lessons L1) -- the same ordering fix the baseline needed.
* **It is stateless across folds.** :meth:`fit` resets; there is nothing to fit.
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
class ReversionReading:
    """Everything computed on one bar, exposed so a run can be audited."""
    close: float
    mean: float
    std: float
    zscore: float               # (close - mean) / std, in return-space
    gap: float                  # mean/close - 1: expected reversion return
    annual_vol: float
    vol_ok: bool
    oversold: bool              # zscore <= -entry_z and vol_ok
    edge: float
    confidence: float


class MeanReversionAgent:
    """Long-only mean reversion: fade dips, exit on reversion or a stop.

    Entry requires *all* of:

    * the close is at least ``entry_z`` standard deviations below its trailing
      mean -- a dip worth fading;
    * trailing annualised volatility at or below ``vol_ceiling`` -- a dip in a
      blowout is not reversion, it is the start of a trend down.

    Exit fires on *any* of:

    * the z-score recovering to ``exit_z`` -- the reversion has played out;
    * volatility above ``vol_ceiling`` -- the regime changed underneath us;
    * the close at or below ``stop_pct`` under the entry mark -- hard stop;
    * the close at or below ``trail_pct`` under the highest close since entry.

    Stops are evaluated on closes, not intrabar lows, so reported drawdowns are
    pessimistic -- the same honest limitation the baseline documents.
    """

    name = "mean_reversion_v1"

    def __init__(
        self,
        lookback: int = 50,
        entry_z: Decimal | str | float = "1.5",
        exit_z: Decimal | str | float = "0.0",
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
        if lookback < 5:
            raise ValueError("lookback must be at least 5 to estimate a mean and std")

        self.lookback = lookback
        self.entry_z = float(to_decimal(entry_z))
        self.exit_z = float(to_decimal(exit_z))
        self.vol_ceiling = to_decimal(vol_ceiling)
        self.stop_pct = to_decimal(stop_pct)
        self.trail_pct = to_decimal(trail_pct)
        self.edge_floor = to_decimal(edge_floor)
        self.edge_ceiling = to_decimal(edge_ceiling)
        self.interval = interval
        self.continuous = continuous
        self.min_annual_vol = to_decimal(min_annual_vol)
        self.spread_bps = to_decimal(spread_bps)
        if name:
            self.name = name

        if self.entry_z <= 0:
            raise ValueError("entry_z must be positive (how far below the mean to buy)")

        self._ann_scale = math.sqrt(periods_per_year(interval, continuous=continuous))
        self._long = False
        self._entry_mark: float | None = None
        self._peak_mark: float | None = None
        self._last_reading: ReversionReading | None = None

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self.lookback + 2

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def last_reading(self) -> ReversionReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lookback": self.lookback,
            "entry_z": str(self.entry_z),
            "exit_z": str(self.exit_z),
            "vol_ceiling": str(self.vol_ceiling),
            "stop_pct": str(self.stop_pct),
            "trail_pct": str(self.trail_pct),
            "interval": self.interval,
            "continuous": self.continuous,
            "warmup_bars": self.warmup_bars,
        }

    # -- strategy protocol -------------------------------------------------

    def fit(self, bars: Sequence[Bar]) -> None:
        self.reset()

    def reset(self) -> None:
        self._long = False
        self._entry_mark = None
        self._peak_mark = None
        self._last_reading = None

    def on_bar(
        self, cursor, instrument: Instrument, mandate: Mandate,
    ) -> Sequence[TradeIntent]:
        history = cursor.history(self.warmup_bars)
        if len(history) < self.warmup_bars:
            return ()

        reading = self._read(history)
        self._last_reading = reading

        # Exits first, before any guard that could suppress them (L1).
        if self._long:
            reason = self._exit_reason(reading)
            if reason is None:
                return ()
            self._long = False
            self._entry_mark = None
            self._peak_mark = None
            return (self._exit_intent(
                instrument, cursor,
                max(to_decimal(reading.annual_vol), self.min_annual_vol), reason,
            ),)

        if reading.annual_vol <= float(self.min_annual_vol):
            # Vol-target and Kelly divide by volatility; refuse to open on a
            # near-zero denominator rather than emit an unbounded position.
            return ()
        if not reading.oversold:
            return ()
        if not mandate.allows(instrument.key):
            return ()
        if mandate.bias_for(instrument.key) < ZERO:
            return ()          # mandate says short; a long-only persona abstains

        self._long = True
        self._entry_mark = reading.close
        self._peak_mark = reading.close
        return (self._entry_intent(
            instrument, cursor, reading, to_decimal(reading.annual_vol),
        ),)

    # -- signal ------------------------------------------------------------

    def _read(self, history: Sequence[Bar]) -> ReversionReading:
        closes = np.array([float(b.close) for b in history], dtype=float)
        close = float(closes[-1])

        window = closes[-self.lookback:]
        mean = float(window.mean())
        std = float(window.std(ddof=1)) if window.size > 1 else 0.0
        zscore = (close - mean) / std if std > 1e-12 else 0.0

        # Expected reversion return: the gap from here back to the mean.
        gap = (mean / close - 1.0) if close > 0 else 0.0

        returns = log_returns(closes)
        annual_vol = float(rolling_std(returns, self.lookback)[-1] * self._ann_scale)

        vol_ok = annual_vol <= float(self.vol_ceiling)
        oversold = zscore <= -self.entry_z and vol_ok

        edge = self._edge(gap)
        confidence = self._confidence(zscore, annual_vol)

        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return ReversionReading(
            close=close, mean=mean, std=std, zscore=zscore, gap=gap,
            annual_vol=annual_vol, vol_ok=vol_ok, oversold=oversold,
            edge=edge, confidence=confidence,
        )

    def _edge(self, gap: float) -> float:
        """Expected forward return = the gap back to the mean, clipped.

        Only a positive gap (price below the mean) is an edge for a long-only
        fader; a negative gap means price is above the mean and there is nothing
        to buy. Clipped for the same reason the baseline clips: Kelly divides by
        variance, so an unclipped extreme would demand the whole book.
        """
        floor, ceiling = float(self.edge_floor), float(self.edge_ceiling)
        return min(ceiling, max(floor, gap))

    def _confidence(self, zscore: float, annual_vol: float) -> float:
        """Conviction in [0,1], rising with the depth of the dip and the calm.

        Anchored at 0.60 so a marginal dip sits below the MODERATE 0.62 floor and
        is filtered by the risk engine, not by a threshold duplicated here.
        """
        # Depth beyond the entry threshold, saturating a couple of sigma past it.
        depth = max(0.0, -zscore - self.entry_z)
        depth_score = min(1.0, depth / 2.0)

        vol_span = float(self.vol_ceiling)
        vol_score = min(1.0, max(0.0, (vol_span - annual_vol) / vol_span)) if vol_span > 0 else 0.0

        blended = 0.65 * depth_score + 0.35 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended))

    def _exit_reason(self, reading: ReversionReading) -> str | None:
        if reading.zscore >= self.exit_z:
            return "reverted"
        if not reading.vol_ok:
            return "vol_spike"
        if self._entry_mark is not None and self._entry_mark > 0:
            if reading.close <= self._entry_mark * (1.0 - float(self.stop_pct)):
                return "hard_stop"
        if self._peak_mark is not None and self._peak_mark > 0:
            if reading.close <= self._peak_mark * (1.0 - float(self.trail_pct)):
                return "trailing_stop"
        return None

    # -- intents -----------------------------------------------------------

    def _entry_intent(
        self, instrument: Instrument, cursor, reading: ReversionReading,
        volatility: Decimal,
    ) -> TradeIntent:
        return TradeIntent(
            instrument=instrument, side=Side.BUY,
            edge=to_decimal(reading.edge), confidence=to_decimal(reading.confidence),
            volatility=volatility, quote=cursor.quote(self.spread_bps),
            strategy=self.name,
        )

    def _exit_intent(
        self, instrument: Instrument, cursor, volatility: Decimal, reason: str,
    ) -> TradeIntent:
        # An exit is a rule firing, not a forecast: full conviction, and a
        # defensive edge large enough to survive even if the engine's reduction
        # path is bypassed and Kelly sizes it (reductions are capped at held qty).
        defensive_edge = max(self.edge_ceiling, volatility * volatility * 4)
        return TradeIntent(
            instrument=instrument, side=Side.SELL,
            edge=defensive_edge, confidence=ONE, volatility=volatility,
            quote=cursor.quote(self.spread_bps), strategy=f"{self.name}:{reason}",
        )


__all__ = ["MeanReversionAgent", "ReversionReading"]
