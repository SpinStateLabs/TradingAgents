"""Feature construction for the regime layer.

Every feature here is **causal**: the value at index ``t`` uses only bars up to
and including ``t``. That is not a stylistic preference. A rolling statistic
that peeks one bar ahead produces a regime signal which appears to anticipate
volatility, and the resulting backtest shows enormous risk-adjusted returns
that vanish the moment it trades live.

The convention is enforced by construction (all windows are trailing) and
checked by :func:`assert_causal`, which is used in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from spintrader.core.types import Bar

# Annualisation factors by bar interval.
PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365, "1w": 52,
}

# Equities trade ~252 sessions a year; crypto trades every day. Using 365 for
# equities overstates annualised volatility by ~20%, which silently shifts
# every vol-targeted position size.
EQUITY_PERIODS_PER_YEAR = {"1d": 252, "1h": 1_638, "1w": 52}


def periods_per_year(interval: str, *, continuous: bool = True) -> float:
    table = PERIODS_PER_YEAR if continuous else EQUITY_PERIODS_PER_YEAR
    try:
        return float(table[interval])
    except KeyError:
        raise ValueError(
            f"no annualisation factor for interval {interval!r}; "
            f"known: {sorted(table)}"
        ) from None


@dataclass(slots=True)
class FeatureMatrix:
    """Causal features aligned to ``timestamps``.

    ``values`` has shape (n_samples, n_features). Row ``i`` is observable at
    ``timestamps[i]`` and no earlier.
    """
    timestamps: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]
    closes: np.ndarray

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def slice_to(self, end_index: int) -> "FeatureMatrix":
        """Rows strictly before ``end_index`` -- the expanding-window fit set."""
        return FeatureMatrix(
            timestamps=self.timestamps[:end_index],
            values=self.values[:end_index],
            names=self.names,
            closes=self.closes[:end_index],
        )


def log_returns(closes: np.ndarray) -> np.ndarray:
    """Log returns, with r[0] = 0 so length is preserved.

    Log rather than simple returns because they are additive across time,
    which matters when the HMM's Gaussian emissions assume roughly symmetric,
    aggregable observations.
    """
    out = np.zeros_like(closes, dtype=float)
    out[1:] = np.diff(np.log(closes))
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation. Index t uses values[t-window+1 : t+1].

    Positions before a full window are filled with the expanding std of what
    is available, never with a future-informed value.

    Vectorised via a sliding-window view over the full-window region, with the
    short warm-up prefix computed directly. It applies ``np.std(ddof=1)`` to
    exactly the same slices as the naive loop, so the result is identical -- but
    without an O(n*window) Python loop, which at minute cadence across thousands
    of bars is the difference between a usable research sweep and an unusable one
    (see lessons L12).
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    out = np.zeros(n, dtype=float)
    # Warm-up: indices with fewer than `window` observations expand. Index 0 is a
    # single value, whose std is 0 -- left as the initialised zero.
    for i in range(1, min(window - 1, n - 1) + 1):
        out[i] = values[:i + 1].std(ddof=1)
    if n >= window:
        win = sliding_window_view(values, window)      # (n-window+1, window)
        out[window - 1:] = win.std(axis=1, ddof=1)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean, same alignment as :func:`rolling_std`. Vectorised identically."""
    values = np.asarray(values, dtype=float)
    n = values.size
    out = np.zeros(n, dtype=float)
    for i in range(min(window - 1, n)):                # expanding warm-up prefix
        out[i] = values[:i + 1].mean()
    if n >= window:
        win = sliding_window_view(values, window)
        out[window - 1:] = win.mean(axis=1)
    return out


def zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing z-score. Zero where the trailing std is degenerate."""
    mean = rolling_mean(values, window)
    std = rolling_std(values, window)
    out = np.zeros_like(values, dtype=float)
    nonzero = std > 1e-12
    out[nonzero] = (values[nonzero] - mean[nonzero]) / std[nonzero]
    return out


def drawdown(closes: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak. Always <= 0."""
    peak = np.maximum.accumulate(closes)
    return (closes - peak) / peak


def build_features(
    bars: Sequence[Bar],
    interval: str = "1d",
    vol_window: int = 20,
    trend_window: int = 50,
    continuous: bool = True,
) -> FeatureMatrix:
    """Build the regime feature matrix from bars.

    Features, all trailing:

    * ``ret``        -- log return of this bar
    * ``vol``        -- annualised trailing realised volatility
    * ``vol_ratio``  -- short vol over long vol; captures vol *regime change*
                        rather than level, which is what distinguishes a
                        quiet market turning turbulent from one already so
    * ``trend``      -- trailing mean return over the trend window
    * ``drawdown``   -- distance below the running peak
    * ``volume_z``   -- trailing z-score of volume

    Volatility and drawdown are the features that actually separate crisis
    from calm; returns alone produce states that mostly track direction and
    flip constantly.
    """
    if len(bars) < max(vol_window, trend_window) + 2:
        raise ValueError(
            f"need at least {max(vol_window, trend_window) + 2} bars to build "
            f"features, got {len(bars)}"
        )

    timestamps = np.array([b.ts for b in bars], dtype=object)
    closes = np.array([float(b.close) for b in bars], dtype=float)
    volumes = np.array([float(b.volume) for b in bars], dtype=float)

    if np.any(closes <= 0):
        raise ValueError("non-positive close price in bar series")

    scale = np.sqrt(periods_per_year(interval, continuous=continuous))
    returns = log_returns(closes)
    vol_short = rolling_std(returns, vol_window) * scale
    vol_long = rolling_std(returns, trend_window) * scale

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_long > 1e-12, vol_short / vol_long, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)

    values = np.column_stack([
        returns,
        vol_short,
        vol_ratio,
        rolling_mean(returns, trend_window),
        drawdown(closes),
        zscore(volumes, vol_window),
    ])

    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop the warm-up region: rows before a full long window are computed
    # from too little data to be comparable, and feeding them to the HMM
    # invents a spurious "early" regime.
    warmup = max(vol_window, trend_window)
    return FeatureMatrix(
        timestamps=timestamps[warmup:],
        values=values[warmup:],
        names=("ret", "vol", "vol_ratio", "trend", "drawdown", "volume_z"),
        closes=closes[warmup:],
    )


def assert_causal(fn, values: np.ndarray, window: int) -> None:
    """Verify a rolling statistic never reads future values.

    Recomputes on a truncated series and checks the overlapping prefix is
    identical. A function that peeks ahead changes its earlier outputs when
    later data is removed.
    """
    full = fn(values, window)
    for cut in (len(values) // 2, len(values) - 1):
        partial = fn(values[:cut], window)
        if not np.allclose(full[:cut], partial, equal_nan=True):
            raise AssertionError(
                f"{fn.__name__} is not causal: truncating at {cut} changed "
                f"earlier outputs, meaning it reads future data"
            )


__all__ = [
    "EQUITY_PERIODS_PER_YEAR", "FeatureMatrix", "PERIODS_PER_YEAR",
    "assert_causal", "build_features", "drawdown", "log_returns",
    "periods_per_year", "rolling_mean", "rolling_std", "zscore",
]
