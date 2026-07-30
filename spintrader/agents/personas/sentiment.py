"""Sentiment persona -- a long-only strategy driven by crowd mood (task 17).

Why this exists
---------------
The price personas (:class:`~spintrader.agents.personas.baseline_trend.BaselineTrendAgent`,
:class:`~spintrader.agents.personas.mean_reversion.MeanReversionAgent`) reason
from one series: price. This one reasons from a *second*, orthogonal series --
the sentiment feed built by :mod:`spintrader.data.sentiment`. Its forecast is
that positive, strengthening crowd sentiment precedes price, and that the mood
decaying is the signal to leave. Whether that edge survives costs is an
empirical question the backtest answers; the persona's job is to express the
thesis causally and let the risk engine decide size.

It is long-only, like the others, because the account is a cash account that
cannot short. It buys optimism, never sells pessimism.

How sentiment reaches it
------------------------
Bars do not carry sentiment, so the series is **injected at construction** as a
mapping ``ts -> score``. The keys are bar close-times: the sentiment feed stamps
each :class:`~spintrader.data.sentiment.SentimentScore` at its bucket's close,
which is the same instant a :class:`~spintrader.core.types.Bar` is stamped, and
the first instant the bucket is fully observable. So :meth:`on_bar` looks the
mood up by the *current* bar's timestamp and reads no future value -- the
lookahead barrier the price cursor enforces is preserved for sentiment by this
alignment. A value can be a :class:`SentimentScore` (carrying mention volume,
used for confidence) or a bare number.

Design constraints it respects (identical to the price personas, on purpose)
----------------------------------------------------------------------------
* **It never sizes.** It reports ``edge``, ``confidence`` and ``volatility``;
  the risk engine decides how much.
* **It never reads the future.** Sentiment is keyed on close-times and the
  slope is measured against a value observed on a prior bar; volatility is
  trailing, over exactly ``warmup_bars``.
* **Exits are evaluated first**, before any guard that could suppress them
  (lessons L1).
* **It is stateless across folds.** :meth:`fit` resets.

Backtest integration is a follow-on
------------------------------------
:func:`spintrader.backtest.engine.run_backtest` feeds a strategy only bars via
the :class:`~spintrader.backtest.engine.ReplayCursor`; there is no sentiment
channel in the cursor today. This persona therefore takes the sentiment series
explicitly at construction and self-aligns it to bar close-times in
:meth:`on_bar`. Wiring sentiment through the backtest/improve loop -- so a
walk-forward run can refetch and re-align it per fold -- is the follow-on task;
see the module ``__doc__`` note in the test file for the concrete seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Sequence

import numpy as np

from spintrader.core.types import Bar, Instrument, Side, ensure_utc, to_decimal
from spintrader.data.sentiment import SentimentScore
from spintrader.quant.features import log_returns, periods_per_year, rolling_std
from spintrader.risk.engine import Mandate, TradeIntent

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class SentimentReading:
    """Everything computed on one bar, exposed so a run can be audited."""
    close: float
    sentiment: float            # current bucket score in [-1, 1] (0 if absent)
    slope: float                # sentiment - last-observed sentiment
    volume: float               # mention engagement backing the current score
    mentions: int
    annual_vol: float
    vol_ok: bool
    has_sentiment: bool         # was a score available for this bar's close?
    bullish: bool               # entry condition satisfied
    edge: float
    confidence: float


class SentimentAgent:
    """Long-only sentiment momentum: buy rising optimism, exit on decay or a stop.

    Entry requires *all* of:

    * a sentiment score exists for this bar's close-time;
    * the score is at or above ``entry_threshold`` -- the crowd is positive;
    * the score is not falling (``slope >= 0``) -- optimism is building, not
      fading;
    * at least ``min_mentions`` mentions back it -- one stray post is not a mood;
    * trailing annualised volatility at or below ``vol_ceiling`` -- a sentiment
      spike inside a blowout is noise, not signal.

    Exit fires on *any* of:

    * the score decaying to ``exit_threshold`` or below (a missing score counts
      as fully decayed) -- the thesis has played out or evaporated;
    * volatility above ``vol_ceiling`` -- the regime changed underneath us;
    * the close at or below ``stop_pct`` under the entry mark -- hard stop;
    * the close at or below ``trail_pct`` under the highest close since entry.

    Stops are evaluated on closes, not intrabar lows, so reported drawdowns are
    pessimistic -- the same honest limitation the price personas document.
    """

    name = "sentiment_v1"

    def __init__(
        self,
        sentiment: Mapping[datetime, "SentimentScore | Decimal | float"] | None = None,
        vol_lookback: int = 20,
        entry_threshold: Decimal | str | float = "0.35",
        exit_threshold: Decimal | str | float = "0.10",
        vol_ceiling: Decimal | str | float = "2.0",
        stop_pct: Decimal | str | float = "0.05",
        trail_pct: Decimal | str | float = "0.08",
        edge_floor: Decimal | str | float = "0.005",
        edge_ceiling: Decimal | str | float = "0.06",
        edge_scale: Decimal | str | float = "0.08",
        level_weight: Decimal | str | float = "0.7",
        slope_weight: Decimal | str | float = "0.3",
        min_mentions: int = 0,
        volume_norm: Decimal | str | float = "25",
        interval: str = "1d",
        continuous: bool = False,
        min_annual_vol: Decimal | str | float = "0.001",
        spread_bps: Decimal | str | float = "5",
        name: str | None = None,
    ) -> None:
        if vol_lookback < 5:
            raise ValueError("vol_lookback must be at least 5 to estimate volatility")

        self.vol_lookback = vol_lookback
        self.entry_threshold = float(to_decimal(entry_threshold))
        self.exit_threshold = float(to_decimal(exit_threshold))
        self.vol_ceiling = to_decimal(vol_ceiling)
        self.stop_pct = to_decimal(stop_pct)
        self.trail_pct = to_decimal(trail_pct)
        self.edge_floor = to_decimal(edge_floor)
        self.edge_ceiling = to_decimal(edge_ceiling)
        self.edge_scale = float(to_decimal(edge_scale))
        self.level_weight = float(to_decimal(level_weight))
        self.slope_weight = float(to_decimal(slope_weight))
        self.min_mentions = int(min_mentions)
        self.volume_norm = float(to_decimal(volume_norm))
        self.interval = interval
        self.continuous = continuous
        self.min_annual_vol = to_decimal(min_annual_vol)
        self.spread_bps = to_decimal(spread_bps)
        if name:
            self.name = name

        if not (-1.0 < self.entry_threshold <= 1.0):
            raise ValueError("entry_threshold must be in (-1, 1]")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be below entry_threshold (a hysteresis band)")

        self._sentiment = self._normalise_sentiment(sentiment or {})
        self._ann_scale = math.sqrt(periods_per_year(interval, continuous=continuous))
        self._long = False
        self._entry_mark: float | None = None
        self._peak_mark: float | None = None
        self._prev_sentiment: float | None = None   # last OBSERVED score, for slope
        self._last_reading: SentimentReading | None = None

    @staticmethod
    def _normalise_sentiment(
        mapping: Mapping[datetime, "SentimentScore | Decimal | float"],
    ) -> dict[datetime, tuple[float, float, int]]:
        """Freeze the injected series into ``ts -> (score, volume, mentions)``.

        Keys are normalised to aware UTC so a lookup by a bar's ``ts`` matches
        regardless of how the caller built the mapping. A bare number carries no
        volume, so its confidence leans entirely on the score's strength.
        """
        out: dict[datetime, tuple[float, float, int]] = {}
        for ts, value in mapping.items():
            key = ensure_utc(ts)
            if isinstance(value, SentimentScore):
                score = float(value.score)
                volume = float(value.volume)
                mentions = int(value.mentions)
            else:
                score = float(to_decimal(value))
                volume = 0.0
                mentions = 0
            out[key] = (max(-1.0, min(1.0, score)), volume, mentions)
        return out

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self.vol_lookback + 2

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def last_reading(self) -> SentimentReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vol_lookback": self.vol_lookback,
            "entry_threshold": str(self.entry_threshold),
            "exit_threshold": str(self.exit_threshold),
            "vol_ceiling": str(self.vol_ceiling),
            "stop_pct": str(self.stop_pct),
            "trail_pct": str(self.trail_pct),
            "min_mentions": self.min_mentions,
            "interval": self.interval,
            "continuous": self.continuous,
            "warmup_bars": self.warmup_bars,
            "sentiment_points": len(self._sentiment),
        }

    # -- strategy protocol -------------------------------------------------

    def fit(self, bars: Sequence[Bar]) -> None:
        self.reset()

    def reset(self) -> None:
        self._long = False
        self._entry_mark = None
        self._peak_mark = None
        self._prev_sentiment = None
        self._last_reading = None

    def on_bar(
        self, cursor, instrument: Instrument, mandate: Mandate,
    ) -> Sequence[TradeIntent]:
        history = cursor.history(self.warmup_bars)
        if len(history) < self.warmup_bars:
            return ()

        reading = self._read(history, cursor.now)
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
        if not reading.has_sentiment:
            return ()          # no mood for this bar; nothing to act on
        if not reading.bullish:
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

    def _lookup(self, ts: datetime) -> tuple[float, float, int, bool]:
        """Sentiment for ``ts``: (score, volume, mentions, present).

        An absent key returns a decayed-neutral score of 0 flagged ``present=False``
        -- enough to trip the decay exit but never to open a position.
        """
        hit = self._sentiment.get(ensure_utc(ts))
        if hit is None:
            return 0.0, 0.0, 0, False
        return hit[0], hit[1], hit[2], True

    def _read(self, history: Sequence[Bar], now: datetime) -> SentimentReading:
        closes = np.array([float(b.close) for b in history], dtype=float)
        close = float(closes[-1])

        sentiment, volume, mentions, present = self._lookup(now)

        # Slope is measured against the last score we actually observed, so a
        # gap in the series does not manufacture a fake swing. Only observed
        # scores update the reference, and only past bars have updated it, so
        # this is causal.
        prior = self._prev_sentiment
        slope = (sentiment - prior) if (present and prior is not None) else 0.0

        returns = log_returns(closes)
        annual_vol = float(rolling_std(returns, self.vol_lookback)[-1] * self._ann_scale)
        vol_ok = annual_vol <= float(self.vol_ceiling)

        bullish = (
            present
            and sentiment >= self.entry_threshold
            and slope >= 0.0
            and mentions >= self.min_mentions
            and vol_ok
        )

        edge = self._edge(sentiment, slope)
        confidence = self._confidence(sentiment, volume)

        if present:
            self._prev_sentiment = sentiment
        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return SentimentReading(
            close=close, sentiment=sentiment, slope=slope, volume=volume,
            mentions=mentions, annual_vol=annual_vol, vol_ok=vol_ok,
            has_sentiment=present, bullish=bullish, edge=edge, confidence=confidence,
        )

    def _edge(self, sentiment: float, slope: float) -> float:
        """Expected forward return from the sentiment level and its slope, clipped.

        Only positive sentiment and positive momentum are an edge for a long-only
        agent; a negative level or a falling mood has nothing to buy. Clipped for
        the same reason the price personas clip -- Kelly divides by variance, so
        an unclipped extreme would demand the whole book.
        """
        raw = self.level_weight * max(0.0, sentiment) + self.slope_weight * max(0.0, slope)
        scaled = raw * self.edge_scale
        floor, ceiling = float(self.edge_floor), float(self.edge_ceiling)
        return min(ceiling, max(floor, scaled))

    def _confidence(self, sentiment: float, volume: float) -> float:
        """Conviction in [0, 1], rising with the strength of the mood and its volume.

        Anchored at 0.60 so a marginal signal sits below the MODERATE 0.62 floor
        and is filtered by the risk engine, not by a threshold duplicated here.
        """
        span = 1.0 - self.entry_threshold
        strength = min(1.0, max(0.0, (sentiment - self.entry_threshold) / span)) if span > 0 else 0.0
        vol_score = min(1.0, max(0.0, volume / self.volume_norm)) if self.volume_norm > 0 else 0.0
        blended = 0.60 * strength + 0.40 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended))

    def _exit_reason(self, reading: SentimentReading) -> str | None:
        if reading.sentiment <= self.exit_threshold:
            return "sentiment_decay"
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
        self, instrument: Instrument, cursor, reading: SentimentReading,
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


__all__ = ["SentimentAgent", "SentimentReading"]
