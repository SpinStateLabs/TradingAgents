"""Performance metrics, with honest error bars.

A Sharpe ratio computed from a short sample and reported as a point estimate is
the most common way a trading system lies to its owner. Two corrections matter
here and both are implemented rather than mentioned:

**Sampling error.** The standard error of an estimated Sharpe ratio is
approximately ``sqrt((1 + SR^2/2) / n)``. On two years of daily data (n=504) a
measured Sharpe of 1.0 carries a standard error near 0.05 annualised... but on
60 observations it is nearer 0.14, and a "Sharpe 1.2" strategy is then
statistically indistinguishable from a "Sharpe 0.8" one. Every Sharpe reported
here travels with its standard error and sample size.

**Selection bias.** This is the one that will actually bite this project. The
self-improvement loop is designed to try many strategy variants and promote the
best. The maximum of N noisy Sharpe estimates is biased upward even when every
variant is worthless -- with 100 trials on 500 observations, the best pure-noise
strategy shows an annualised Sharpe near 0.9. The Deflated Sharpe Ratio
(Bailey & Lopez de Prado) adjusts the significance threshold for the number of
trials, so a variant must beat what noise alone would have produced. Any
promotion decision that ignores ``n_trials`` is mining, not research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import numpy as np

TRADING_DAYS = 252
CRYPTO_DAYS = 365


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, via erf so scipy is not required."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over (0,1), which is far beyond what is needed to set a
    significance threshold, and avoids a scipy dependency in the hot path.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0,1), got {p}")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@dataclass(slots=True)
class Scorecard:
    """Performance summary for one backtest or live period."""

    # provenance
    n_observations: int
    periods_per_year: int
    start: datetime | None = None
    end: datetime | None = None
    n_trials: int = 1                 # how many variants were tried to get here

    # returns
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0

    # risk-adjusted, with error bars
    sharpe: float = 0.0
    sharpe_stderr: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # drawdown
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0

    # trade statistics
    n_trades: int = 0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0

    # costs
    fees_paid: float = 0.0
    cost_drag: float = 0.0            # fees as a fraction of gross P&L

    # significance
    psr: float = 0.0                  # probabilistic Sharpe ratio
    deflated_sharpe: float = 0.0      # multiple-testing adjusted
    skew: float = 0.0
    excess_kurtosis: float = 0.0

    # benchmark
    benchmark_return: float | None = None
    alpha: float | None = None

    @property
    def sharpe_ci95(self) -> tuple[float, float]:
        """95% confidence interval on the Sharpe estimate."""
        half = 1.96 * self.sharpe_stderr
        return (self.sharpe - half, self.sharpe + half)

    @property
    def is_significant(self) -> bool:
        """Whether performance survives multiple-testing adjustment.

        The threshold is the Deflated Sharpe Ratio rather than the raw Sharpe
        or even the PSR, because the loop that produced this result was
        searching. A DSR below 0.95 means the result is within what noise plus
        selection would have produced.
        """
        return self.deflated_sharpe >= 0.95

    def verdict(self) -> str:
        low, high = self.sharpe_ci95
        parts = [
            f"Sharpe {self.sharpe:.2f} (95% CI {low:.2f}..{high:.2f}, n={self.n_observations})",
            f"CAGR {self.cagr:.1%}",
            f"maxDD {self.max_drawdown:.1%}",
            f"cost drag {self.cost_drag:.1%}",
        ]
        if self.n_trials > 1:
            parts.append(f"DSR {self.deflated_sharpe:.2f} over {self.n_trials} trials")
        parts.append("SIGNIFICANT" if self.is_significant else "not significant")
        return " | ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Deepest peak-to-trough decline and its duration in observations."""
    if equity.size == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(equity)
    # Guard against a zero or negative peak, which a blown-up account produces.
    safe = np.where(peaks > 0, peaks, np.nan)
    drawdowns = (equity - peaks) / safe
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)
    trough = int(np.argmin(drawdowns))
    depth = float(drawdowns[trough])

    # Duration: from the peak preceding the trough to recovery (or series end).
    peak_index = int(np.argmax(equity[:trough + 1])) if trough > 0 else 0
    recovery = trough
    for i in range(trough, equity.size):
        if equity[i] >= equity[peak_index]:
            recovery = i
            break
    else:
        recovery = equity.size - 1
    return depth, int(recovery - peak_index)


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------
#
# A note on units, because getting this wrong is silent and severe.
#
# The PSR and DSR formulae are defined on the **per-period** Sharpe ratio -- the
# one computed from raw observation-frequency returns, with no sqrt(252)
# applied. Feeding them an annualised Sharpe inflates the test statistic by
# sqrt(periods_per_year), which for daily data is ~15.9x, and makes essentially
# every strategy look overwhelmingly significant. That is precisely the
# comforting lie this module exists to prevent.
#
# These functions therefore take the annualised Sharpe plus
# ``periods_per_year`` and de-annualise internally, so a caller cannot supply
# the wrong one by accident.


def _deannualise(sharpe_annual: float, periods_per_year: int) -> float:
    return sharpe_annual / math.sqrt(periods_per_year) if periods_per_year > 0 else 0.0


def sharpe_standard_error(
    sharpe: float, n: int, periods_per_year: int = 1,
) -> float:
    """Standard error of an estimated Sharpe ratio.

    Lo (2002) approximation for iid returns: ``sqrt((1 + SR^2/2) / n)`` on the
    per-period Sharpe. When ``periods_per_year`` is supplied, ``sharpe`` is
    treated as annualised and the returned error is annualised to match, so the
    estimate and its error are always in the same units.

    Returns are not truly iid, which makes this optimistic -- a floor on the
    uncertainty, not a ceiling.
    """
    if n <= 1:
        return float("inf")
    per_period = _deannualise(sharpe, periods_per_year) if periods_per_year > 1 else sharpe
    se_per_period = math.sqrt((1.0 + 0.5 * per_period * per_period) / n)
    return se_per_period * math.sqrt(periods_per_year) if periods_per_year > 1 else se_per_period


def probabilistic_sharpe(
    sharpe: float, n: int, skew: float, excess_kurtosis: float,
    benchmark: float = 0.0, periods_per_year: int = 1,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark``.

    ``sharpe`` and ``benchmark`` are annualised when ``periods_per_year`` > 1.

    Skew and kurtosis matter here: a strategy that sells options or fades
    volatility has left-skewed, fat-tailed returns, and its raw Sharpe
    overstates its quality precisely because the bad outcome has not happened
    yet. Both reduce the probability.
    """
    if n <= 1:
        return 0.0

    sr = _deannualise(sharpe, periods_per_year) if periods_per_year > 1 else sharpe
    bench = _deannualise(benchmark, periods_per_year) if periods_per_year > 1 else benchmark

    # The formula uses FULL kurtosis (3 for a normal), not excess. Using excess
    # here flips the sign of the SR^2 term and drives the denominator negative
    # at moderate Sharpe values.
    kurtosis = excess_kurtosis + 3.0

    denominator = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr * sr
    if denominator <= 0:
        # Degenerate higher moments; no defensible probability statement.
        return 0.0

    statistic = (sr - bench) * math.sqrt(n - 1) / math.sqrt(denominator)
    return _norm_cdf(statistic)


def deflated_sharpe(
    sharpe: float, n: int, skew: float, excess_kurtosis: float,
    n_trials: int, trial_sharpe_variance: float | None = None,
    periods_per_year: int = 1,
) -> float:
    """Sharpe significance adjusted for the number of trials.

    The maximum of N noisy Sharpe estimates is biased upward even when every
    variant is worthless. This computes the expected maximum under the null and
    uses it as the benchmark, so a candidate must beat what selection alone
    would have produced.

    ``trial_sharpe_variance`` is the variance of per-period Sharpe estimates
    across trials. When unknown, the sampling variance is used as a
    conservative stand-in.
    """
    if n_trials <= 1:
        return probabilistic_sharpe(
            sharpe, n, skew, excess_kurtosis, benchmark=0.0,
            periods_per_year=periods_per_year,
        )

    per_period = _deannualise(sharpe, periods_per_year) if periods_per_year > 1 else sharpe
    variance = (
        trial_sharpe_variance
        if trial_sharpe_variance is not None
        else (1.0 + 0.5 * per_period * per_period) / n
    )
    if variance <= 0:
        return probabilistic_sharpe(sharpe, n, skew, excess_kurtosis,
                                    periods_per_year=periods_per_year)

    # Expected maximum of N iid standard normals (Euler-Mascheroni expansion).
    gamma = 0.5772156649015329
    term = (1.0 - gamma) * _norm_ppf(1.0 - 1.0 / n_trials) + \
        gamma * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    expected_max_per_period = math.sqrt(variance) * term

    # Re-annualise the benchmark so both arguments share the caller's units.
    benchmark = (
        expected_max_per_period * math.sqrt(periods_per_year)
        if periods_per_year > 1 else expected_max_per_period
    )
    return probabilistic_sharpe(
        sharpe, n, skew, excess_kurtosis, benchmark=benchmark,
        periods_per_year=periods_per_year,
    )


def score(
    equity_curve: Sequence[float] | np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    n_trades: int = 0,
    trade_pnls: Sequence[float] | None = None,
    fees_paid: float = 0.0,
    turnover: float = 0.0,
    benchmark_curve: Sequence[float] | np.ndarray | None = None,
    n_trials: int = 1,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Scorecard:
    """Compute a full scorecard from an equity curve."""
    equity = np.asarray(equity_curve, dtype=float)

    # Deterministic quantities are computed whenever they are defined, even on a
    # sample too short for a Sharpe. Returning a wholly empty card in that case
    # discards perfectly good information -- total return, fees and the
    # benchmark comparison do not need a distribution.
    total_return = (
        float(equity[-1] / equity[0] - 1.0)
        if equity.size >= 2 and equity[0] > 0 else 0.0
    )

    benchmark_return = None
    alpha = None
    if benchmark_curve is not None:
        bench = np.asarray(benchmark_curve, dtype=float)
        if bench.size >= 2 and bench[0] > 0:
            benchmark_return = float(bench[-1] / bench[0] - 1.0)
            alpha = total_return - benchmark_return

    gross_pnl = (
        abs(float(equity[-1] - equity[0])) + abs(fees_paid)
        if equity.size >= 2 else abs(fees_paid)
    )
    cost_drag = (abs(fees_paid) / gross_pnl) if gross_pnl > 0 else 0.0

    hit_rate = 0.0
    profit_factor = 0.0
    if trade_pnls:
        pnls = np.asarray(list(trade_pnls), dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        hit_rate = float(wins.size / pnls.size) if pnls.size else 0.0
        gross_loss = float(-losses.sum())
        profit_factor = (
            float(wins.sum() / gross_loss) if gross_loss > 0
            else (float("inf") if wins.size else 0.0)
        )

    def partial(n_obs: int) -> Scorecard:
        """Card with only the deterministic fields populated."""
        return Scorecard(
            n_observations=n_obs, periods_per_year=periods_per_year,
            start=start, end=end, n_trials=n_trials,
            total_return=total_return, n_trades=n_trades,
            hit_rate=hit_rate, profit_factor=profit_factor, turnover=turnover,
            fees_paid=fees_paid, cost_drag=cost_drag,
            benchmark_return=benchmark_return, alpha=alpha,
        )

    if equity.size < 2:
        return partial(int(equity.size))

    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    n = returns.size
    if n < 2:
        # One return cannot yield a volatility, so no risk-adjusted figure is
        # defensible -- but the deterministic fields above still are.
        return partial(n)

    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    years = n / periods_per_year
    cagr = (
        float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
        if years > 0 and equity[0] > 0 and equity[-1] > 0 else 0.0
    )

    volatility = std * math.sqrt(periods_per_year)
    sharpe = (mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = (mean / downside_std * math.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    depth, duration = max_drawdown(equity)
    calmar = (cagr / abs(depth)) if depth < 0 else 0.0

    # Higher moments, needed for PSR/DSR.
    centred = returns - mean
    skew = float((centred ** 3).mean() / std ** 3) if std > 0 else 0.0
    excess_kurtosis = float((centred ** 4).mean() / std ** 4 - 3.0) if std > 0 else 0.0

    stderr = sharpe_standard_error(sharpe, n, periods_per_year)
    psr = probabilistic_sharpe(sharpe, n, skew, excess_kurtosis,
                               periods_per_year=periods_per_year)
    dsr = deflated_sharpe(sharpe, n, skew, excess_kurtosis, n_trials,
                          periods_per_year=periods_per_year)

    return Scorecard(
        n_observations=n,
        periods_per_year=periods_per_year,
        start=start, end=end, n_trials=n_trials,
        total_return=total_return, cagr=cagr, volatility=volatility,
        sharpe=sharpe, sharpe_stderr=stderr, sortino=sortino, calmar=calmar,
        max_drawdown=depth, max_drawdown_days=duration,
        n_trades=n_trades, hit_rate=hit_rate, profit_factor=profit_factor,
        turnover=turnover, fees_paid=fees_paid, cost_drag=cost_drag,
        psr=psr, deflated_sharpe=dsr, skew=skew, excess_kurtosis=excess_kurtosis,
        benchmark_return=benchmark_return, alpha=alpha,
    )


__all__ = [
    "CRYPTO_DAYS", "Scorecard", "TRADING_DAYS", "deflated_sharpe",
    "max_drawdown", "probabilistic_sharpe", "score", "sharpe_standard_error",
]
