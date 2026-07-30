"""Tail-fragility gauge -- a "black swan" indicator that is honest about itself.

A black swan is, by construction, unpredictable: if it could be forecast from
the trailing tape it would not be one. This module therefore does **not** claim
to predict crashes. It measures how *fragile* current conditions are -- whether
the ground is soft underfoot -- by reading the properties of a market that tend
to precede large moves without ever timing them:

* **Fat, left-skewed returns.** A distribution with heavy tails and negative
  skew has more mass where the ruinous outcomes live. This is a statement about
  the shape of recent returns, not a prediction of the next one.
* **A vol regime turning.** ``vol_ratio`` (short realised vol over long) rising
  through 1 is a quiet market becoming turbulent -- the transition, not the
  level, is what distinguishes fragile-and-building from already-blown-out.
* **Distance below the peak.** Drawdown is coincident, not leading: it says the
  slide is already underway. It earns a small weight precisely because it is the
  one component that cannot cry wolf -- if it is high, something is already
  happening.
* **An EVT tail estimate.** Fitting a generalised Pareto to peaks-over-threshold
  of the loss tail reads the *asymptotic* tail heaviness (the GPD shape) and a
  conditional-VaR-style expected shortfall -- the honest way to talk about tail
  risk from a finite sample, rather than trusting the empirical worst day.
* **Optionally the HMM crisis posterior.** When a fitted causal regime model is
  supplied, its filtered probability of the most-stressed state is folded in.

The output is a single ``black_swan_score`` in ``[0, 1]``. It is a gauge, not a
signal, and it must be judged by its false-alarm rate (see
:func:`evaluate_false_alarms`) -- a fragility reading that fires constantly and
is rarely followed by anything is a broken gauge, however alarming it looks.

Everything here is **causal**: every component reads a trailing window only, so
the score at ``t`` uses no data after ``t``. There are no smoothed HMM states
(those are lookahead; see :mod:`spintrader.quant.regime`). All arithmetic is
float -- this is a *model* whose output never touches the ledger, exactly like
:class:`spintrader.venues.paper.SlippageModel`; the one value that reaches
money-adjacent code (the mandate's ``regime_risk``) is handed back as ``Decimal``
by :func:`combine_regime_risk`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from spintrader.core.types import to_decimal
from spintrader.quant.features import drawdown as _drawdown
from spintrader.quant.features import log_returns, rolling_std

log = logging.getLogger(__name__)

# The component keys, fixed so the audit trail and the weights never drift apart.
COMPONENTS = ("kurtosis", "skew", "vol_ratio", "drawdown", "evt", "regime")


def _saturate(value: float, reference: float) -> float:
    """Map a non-negative magnitude into ``[0, 1)`` with a soft knee.

    ``value == reference`` maps to 0.5 and the response saturates thereafter, so
    a single extreme reading cannot dominate the blend the way a linear scaling
    would. Negative or non-finite inputs map to 0 -- the benign end -- because a
    component that could not be measured must not manufacture fragility.
    """
    if not math.isfinite(value) or value <= 0.0 or reference <= 0.0:
        return 0.0
    return value / (value + reference)


def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Sample skew and **excess** kurtosis of ``returns``.

    Kept byte-for-byte identical to the computation in
    :func:`spintrader.backtest.scorecard.score` so the fragility gauge and the
    performance report describe the same distribution the same way -- ``std`` via
    ``ddof=1``, third/fourth standardised central moments. Note the scorecard's
    PSR/DSR path adds 3 back to get **full** kurtosis; the gauge keeps excess,
    because "how much fatter than normal" is the quantity that means fragility.
    """
    n = returns.size
    if n < 2:
        return 0.0, 0.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std <= 0.0:
        return 0.0, 0.0
    centred = returns - mean
    skew = float((centred ** 3).mean() / std ** 3)
    excess_kurtosis = float((centred ** 4).mean() / std ** 4 - 3.0)
    return skew, excess_kurtosis


@dataclass(slots=True)
class EVTFit:
    """A peaks-over-threshold generalised-Pareto fit of the loss tail.

    ``shape`` is the GPD shape parameter (``xi``): > 0 is a genuinely heavy,
    power-law tail; ~0 is exponential; < 0 is a bounded tail. ``cvar`` is the
    expected shortfall at ``var_level`` -- the average loss *conditional on* being
    beyond the VaR -- expressed as a positive fractional loss. ``ok`` is False
    when the window was too short or too degenerate to fit, in which case the
    component is dropped from the blend rather than guessed at.
    """
    ok: bool
    shape: float = float("nan")           # xi
    scale: float = float("nan")           # beta
    threshold: float = float("nan")       # u, the POT threshold (a loss level)
    n_exceedances: int = 0
    var: float = float("nan")             # value-at-risk at var_level, as a loss
    cvar: float = float("nan")            # expected shortfall at var_level


@dataclass(slots=True)
class TailReading:
    """Every input, sub-score and weight behind one ``black_swan_score``.

    Exposed in full because a risk gauge that cannot be interrogated is a risk
    in itself: when the score is high the operator must be able to see *why* --
    fat tails, a turning vol regime, a deep drawdown, or the EVT tail -- and when
    it is a false alarm, which component cried wolf.
    """
    score: float
    # Raw readings (the distribution, not the score).
    excess_kurtosis: float
    skew: float
    vol_ratio: float
    drawdown: float                       # <= 0, distance below running peak
    evt: EVTFit
    crisis_posterior: float | None        # HMM most-stressed-state mass, if given
    n: int                                # window length actually used
    # Per-component sub-scores in [0,1] and the renormalised weights applied to
    # the components that were measurable this bar.
    subscores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "excess_kurtosis": self.excess_kurtosis,
            "skew": self.skew,
            "vol_ratio": self.vol_ratio,
            "drawdown": self.drawdown,
            "evt_shape": self.evt.shape,
            "evt_cvar": self.evt.cvar,
            "evt_ok": self.evt.ok,
            "crisis_posterior": self.crisis_posterior,
            "n": self.n,
            "subscores": dict(self.subscores),
            "weights": dict(self.weights),
        }


@dataclass(slots=True)
class TailRiskGauge:
    """A causal 0..1 tail-fragility gauge over a trailing window.

    The reference constants set where each component reads "half fragile"; they
    are deliberately calibrated to daily equity/crypto returns and are the knobs
    to turn per asset class. The default weights lean on the EVT tail (the only
    component built specifically to quantify tail risk) and give drawdown the
    least, since it is coincident rather than leading.
    """

    window: int = 250
    vol_short: int = 20
    vol_long: int = 100
    # EVT peaks-over-threshold configuration.
    evt_quantile: float = 0.90            # threshold = this quantile of losses
    min_exceedances: int = 15             # below this, the GPD fit is not trusted
    var_level: float = 0.99              # tail probability for VaR / expected shortfall
    # Reference magnitudes for the soft-knee mappings.
    kurtosis_ref: float = 3.0            # excess kurtosis of 3 -> 0.5
    skew_ref: float = 1.0                # left-skew of -1 -> 0.5
    vol_ratio_ref: float = 0.5           # short vol 1.5x long -> 0.5
    drawdown_ref: float = 0.10           # 10% below peak -> 0.5
    cvar_ref: float = 0.05               # 5% expected shortfall -> 0.5
    shape_ref: float = 0.3               # GPD shape 0.3 -> 0.5
    # Expected shortfall is the robust tail signal (it already blends shape and
    # scale into a concrete loss); the GPD shape is a noisier secondary read, so
    # it gets the minority of the EVT weight and can lift the sub-score but not
    # sink a genuinely large shortfall on a negative shape estimate.
    cvar_weight: float = 0.7
    # Minimum returns needed before the moment components are believed.
    min_moment_obs: int = 20
    weights: dict[str, float] = field(default_factory=lambda: {
        "kurtosis": 0.15,
        "skew": 0.15,
        "vol_ratio": 0.15,
        "drawdown": 0.10,
        "evt": 0.30,
        "regime": 0.15,
    })

    # -- EVT ---------------------------------------------------------------

    def fit_evt(self, returns: np.ndarray) -> EVTFit:
        """Fit a GPD to peaks-over-threshold of the **loss** tail.

        Losses are ``-returns``, so a drop is a positive exceedance. The
        threshold is the ``evt_quantile`` of the losses; the GPD is fitted to the
        excesses above it with the location pinned to zero (the POT convention).
        From the fit we read the tail shape and a conditional-VaR expected
        shortfall via the standard McNeil-Frey-Embrechts closed forms.

        Degrades gracefully to ``EVTFit(ok=False)`` -- never raises -- when the
        window is too short, the losses are degenerate (a flat or one-sided
        series yields too few exceedances), or SciPy's optimiser fails. A gauge
        that blows up on a quiet week is worse than one that admits it cannot see
        the tail yet.
        """
        losses = -np.asarray(returns, dtype=float)
        losses = losses[np.isfinite(losses)]
        n = losses.size
        if n < max(self.min_exceedances * 3, 30):
            return EVTFit(ok=False)

        threshold = float(np.quantile(losses, self.evt_quantile))
        excesses = losses[losses > threshold] - threshold
        n_exc = int(excesses.size)
        # Too few exceedances, or the threshold sits on a spike of identical
        # values so there is no spread to fit -- either way the shape is noise.
        if n_exc < self.min_exceedances or float(excesses.std()) <= 1e-12:
            return EVTFit(ok=False)

        try:
            from scipy.stats import genpareto
            shape, _loc, scale = genpareto.fit(excesses, floc=0.0)
        except Exception as exc:                       # pragma: no cover - optimiser edge
            log.debug("EVT genpareto fit failed: %s", exc)
            return EVTFit(ok=False)

        if not (math.isfinite(shape) and math.isfinite(scale)) or scale <= 0.0:
            return EVTFit(ok=False)

        var, cvar = self._tail_measures(shape, scale, threshold, n, n_exc)
        return EVTFit(
            ok=True, shape=float(shape), scale=float(scale), threshold=threshold,
            n_exceedances=n_exc, var=var, cvar=cvar,
        )

    def _tail_measures(
        self, shape: float, scale: float, threshold: float, n: int, n_exc: int,
    ) -> tuple[float, float]:
        """VaR and expected shortfall at ``var_level`` from a POT-GPD fit.

        Uses the tail estimator ``P(L > x) = (n_exc/n)(1 + xi (x-u)/beta)^(-1/xi)``
        inverted at the tail probability, then the closed-form expected shortfall.
        For ``xi >= 1`` the theoretical mean is infinite; the shape is clamped for
        the ES denominator alone (VaR and the reported shape stay exact) so the
        number stays finite and the sub-score simply saturates toward 1.
        """
        p_tail = 1.0 - self.var_level
        ratio = (n / n_exc) * p_tail                    # < 1 for a sensible level
        if abs(shape) < 1e-6:
            var = threshold - scale * math.log(ratio)
        else:
            var = threshold + (scale / shape) * (ratio ** (-shape) - 1.0)

        xi_es = min(shape, 0.99)                         # keep ES finite if xi>=1
        cvar = (var + scale - shape * threshold) / (1.0 - xi_es)
        # Losses below the threshold cannot produce a tail VaR below it.
        var = max(var, threshold)
        cvar = max(cvar, var)
        return float(var), float(cvar)

    # -- the reading -------------------------------------------------------

    def reading(
        self,
        closes: np.ndarray,
        *,
        crisis_posterior: float | None = None,
        regime_model: object | None = None,
        features: object | None = None,
    ) -> TailReading:
        """Compute a full :class:`TailReading` from a trailing window of closes.

        Only the last ``window`` closes are used, so calling this on a longer
        series and on any right-truncation of it that still ends at the same bar
        yields the same reading -- causality by construction.

        The HMM crisis posterior may be supplied three ways, in priority order:
        an explicit ``crisis_posterior`` float; or a fitted ``regime_model`` plus
        the aligned ``features`` matrix, from which the filtered (causal-forward)
        probability of the most-stressed state is read. If the regime cannot be
        assessed the component is simply absent -- the gauge composes with the HMM
        but never depends on it.
        """
        closes = np.asarray(closes, dtype=float)
        closes = closes[-self.window:] if closes.size > self.window else closes
        n = int(closes.size)

        if crisis_posterior is None and regime_model is not None and features is not None:
            crisis_posterior = _safe_crisis_posterior(regime_model, features)

        if n < 2:
            # Nothing measurable; a benign, explicitly-empty reading.
            evt = EVTFit(ok=False)
            return TailReading(
                score=0.0, excess_kurtosis=0.0, skew=0.0, vol_ratio=1.0,
                drawdown=0.0, evt=evt, crisis_posterior=crisis_posterior, n=n,
            )

        returns = log_returns(closes)[1:]                # drop the seeded r[0]=0
        skew, excess_kurtosis = _moments(returns)
        vol_ratio = self._vol_ratio(returns)
        current_dd = float(_drawdown(closes)[-1])        # <= 0, distance below peak
        evt = self.fit_evt(returns)

        subscores: dict[str, float] = {}
        # (a) fat tails and left skew -- only once there are enough returns for
        # the fourth moment to be anything but noise.
        if returns.size >= self.min_moment_obs:
            subscores["kurtosis"] = _saturate(excess_kurtosis, self.kurtosis_ref)
            subscores["skew"] = _saturate(-skew, self.skew_ref)   # left tail only
        # (b) a vol regime turning up.
        if vol_ratio is not None:
            subscores["vol_ratio"] = _saturate(vol_ratio - 1.0, self.vol_ratio_ref)
        # (c) drawdown from the running peak (coincident).
        subscores["drawdown"] = _saturate(-current_dd, self.drawdown_ref)
        # (d) EVT tail: expected shortfall (robust) plus the heaviness bonus.
        if evt.ok:
            subscores["evt"] = self.cvar_weight * _saturate(evt.cvar, self.cvar_ref) \
                + (1.0 - self.cvar_weight) * _saturate(max(0.0, evt.shape), self.shape_ref)
        # (e) HMM crisis posterior, if available.
        if crisis_posterior is not None and math.isfinite(crisis_posterior):
            subscores["regime"] = float(np.clip(crisis_posterior, 0.0, 1.0))

        score, applied = self._blend(subscores)
        return TailReading(
            score=score,
            excess_kurtosis=excess_kurtosis,
            skew=skew,
            vol_ratio=vol_ratio if vol_ratio is not None else 1.0,
            drawdown=current_dd,
            evt=evt,
            crisis_posterior=crisis_posterior,
            n=n,
            subscores=subscores,
            weights=applied,
        )

    def black_swan_score(
        self,
        closes: np.ndarray,
        *,
        crisis_posterior: float | None = None,
        regime_model: object | None = None,
        features: object | None = None,
    ) -> float:
        """The scalar tail-fragility score in ``[0, 1]``. See :meth:`reading`."""
        return self.reading(
            closes, crisis_posterior=crisis_posterior,
            regime_model=regime_model, features=features,
        ).score

    def series(self, closes: np.ndarray) -> np.ndarray:
        """The score at every bar, each from its own trailing window.

        ``series(closes)[t] == reading(closes[:t+1]).score``, which is the
        property the causality test checks: truncating the future cannot move a
        past score. Computed the honest, non-clever way (one reading per bar) so
        there is no shared state across bars to leak information backward.
        """
        closes = np.asarray(closes, dtype=float)
        out = np.zeros(closes.size, dtype=float)
        for t in range(closes.size):
            out[t] = self.reading(closes[: t + 1]).score
        return out

    # -- internals ---------------------------------------------------------

    def _vol_ratio(self, returns: np.ndarray) -> float | None:
        """Short realised vol over long, or ``None`` when the long window is unmet.

        Reuses :func:`spintrader.quant.features.rolling_std` and reads only the
        final value, so it is the same trailing statistic the feature layer and
        the HMM see. The annualisation factor cancels in the ratio, so raw
        per-bar std is used. Returns ``None`` (component absent) rather than a
        warmed-up-from-too-little-data number when there are fewer returns than
        the long window -- an unreliable ratio should abstain, not mislead.
        """
        if returns.size < self.vol_long:
            return None
        short = float(rolling_std(returns, self.vol_short)[-1])
        long = float(rolling_std(returns, self.vol_long)[-1])
        if long <= 1e-12:
            return None
        return short / long

    def _blend(self, subscores: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Weighted mean over the components that were measurable this bar.

        Weights are renormalised over present components, so a missing EVT fit or
        an absent regime model reweights the survivors rather than silently
        dragging the score toward zero. Returns the score clipped to ``[0, 1]``
        and the weights actually applied, for the audit trail.
        """
        present = {k: self.weights[k] for k in subscores if self.weights.get(k, 0.0) > 0.0}
        total = sum(present.values())
        if total <= 0.0:
            return 0.0, {}
        applied = {k: w / total for k, w in present.items()}
        score = sum(subscores[k] * w for k, w in applied.items())
        return float(np.clip(score, 0.0, 1.0)), applied


def _safe_crisis_posterior(regime_model: object, features: object) -> float | None:
    """Filtered probability of the most-stressed regime state, or ``None``.

    Uses :meth:`RegimeModel.filter_latest` -- the causal-forward filter, never the
    smoothed Viterbi states, which would be lookahead. The most-stressed state is
    the last in canonical (calm-first) order, so its posterior mass is the
    crisis probability. Any failure (model not fitted, ``hmmlearn`` absent, shape
    mismatch) yields ``None`` so the gauge degrades to its market-data components.
    """
    try:
        state = regime_model.filter_latest(features)      # type: ignore[attr-defined]
        probabilities = np.asarray(state.probabilities, dtype=float)
        if probabilities.size == 0:
            return None
        return float(probabilities[-1])
    except Exception as exc:                               # pragma: no cover - defensive
        log.debug("crisis posterior unavailable: %s", exc)
        return None


# --------------------------------------------------------------------------
# (1) Feeding fragility into the mandate's regime_risk
# --------------------------------------------------------------------------

def combine_regime_risk(
    regime_risk: Decimal,
    fragility: float,
    *,
    weight: float = 0.5,
) -> Decimal:
    """Fold a fragility score into the HMM's ``regime_risk``, returning ``Decimal``.

    The mandate already scales exposure down with ``regime_risk`` (see
    :meth:`RiskProfile.scaled_for_regime`), so the cheapest way to make fragility
    *act* is to raise that scalar. This uses a noisy-OR,
    ``1 - (1 - regime_risk)(1 - weight * fragility)``, which has three properties
    that matter for composing with the HMM:

    * it never *lowers* the HMM's risk -- fragility can only add caution, never
      talk the system back into exposure the regime model just removed;
    * it stays in ``[0, 1]`` for any inputs in range, so it drops straight into
      the existing exposure-scaling path;
    * ``weight`` (<= 1) caps how much a fragility reading alone is allowed to
      tighten, so a jumpy gauge cannot flatten the book on its own.

    Returned as ``Decimal`` because the result reaches money-adjacent sizing;
    the float arithmetic stays inside this model, matching the house rule that
    only values touching the ledger are ``Decimal`` (cf. ``assess_regime``).
    """
    hmm = max(0.0, min(1.0, float(regime_risk)))
    frag = max(0.0, min(1.0, fragility))
    w = max(0.0, min(1.0, weight))
    combined = 1.0 - (1.0 - hmm) * (1.0 - w * frag)
    return to_decimal(max(0.0, min(1.0, combined)))


# --------------------------------------------------------------------------
# (2) False-alarm evaluation -- judge the gauge, do not celebrate it
# --------------------------------------------------------------------------

@dataclass(slots=True)
class FalseAlarmReport:
    """What a fragility threshold actually bought, over a real bar series.

    A tail gauge earns its keep only if forward outcomes are meaningfully worse
    when it is high than when it is low -- *and* it must be read alongside how
    often it fires, because a gauge that is "high" 40% of the time and is right
    a little more than half of those has mostly produced false alarms. Every
    field here exists so that trade-off is visible rather than assumed.
    """
    threshold: float
    horizon: int
    n: int                               # bars with a full forward window
    alarm_rate: float                    # fraction of bars scoring >= threshold
    n_high: int
    n_low: int
    # Forward drawdown (most negative forward return over the horizon), by bucket.
    mean_fdd_high: float
    mean_fdd_low: float
    median_fdd_high: float
    median_fdd_low: float
    worst_fdd_high: float
    worst_fdd_low: float
    # P(forward drawdown breaches -tail_move) in each bucket -- the "hit rate"
    # and the "false-alarm rate" of the gauge at this threshold.
    tail_move: float
    hit_rate_high: float                 # P(big drop | high score) -- true positives
    false_alarm_rate_high: float         # P(no big drop | high score) -- false alarms
    base_rate: float                     # P(big drop) unconditionally

    def summary(self) -> str:
        lift = (self.mean_fdd_high / self.mean_fdd_low
                if self.mean_fdd_low < 0 else float("nan"))
        return (
            f"threshold={self.threshold:.2f} horizon={self.horizon}d | "
            f"fires {self.alarm_rate:.1%} of bars ({self.n_high}/{self.n}) | "
            f"mean fwd drawdown high {self.mean_fdd_high:.2%} vs low "
            f"{self.mean_fdd_low:.2%} (x{lift:.1f}) | "
            f"worst high {self.worst_fdd_high:.2%} vs low {self.worst_fdd_low:.2%} | "
            f"P(drop<{-self.tail_move:.0%}) base {self.base_rate:.1%}, "
            f"given-high {self.hit_rate_high:.1%} "
            f"=> false alarms {self.false_alarm_rate_high:.1%} of alarms"
        )


def forward_drawdown(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Worst forward return over the next ``horizon`` bars, per bar.

    ``fdd[t] = min(0, min_{t < k <= t+horizon} closes[k] / closes[t] - 1)`` -- the
    deepest drop below the entry an position at ``t`` would have sat through over
    the horizon, clamped at 0 so a path that only rose is a non-event rather than
    a spurious "gain". Always <= 0. The trailing ``horizon`` bars have no full
    forward window and are returned as ``nan`` so callers exclude them rather
    than scoring them against a truncated future.
    """
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    out = np.full(n, np.nan, dtype=float)
    for t in range(n - 1):
        end = min(t + horizon, n - 1)
        if end <= t:
            continue
        window = closes[t + 1: end + 1]
        out[t] = min(0.0, float(window.min() / closes[t] - 1.0))
    if n - horizon < n:
        out[max(0, n - horizon):] = np.nan
    return out


def evaluate_false_alarms(
    closes: np.ndarray,
    gauge: TailRiskGauge | None = None,
    *,
    threshold: float = 0.6,
    horizon: int = 21,
    tail_move: float = 0.10,
    scores: np.ndarray | None = None,
) -> FalseAlarmReport:
    """Score a real series and measure what a fragility threshold actually predicts.

    Splits bars into high- (``score >= threshold``) and low-fragility buckets and
    compares their forward-drawdown distributions, plus the conditional
    probability of a ``tail_move`` drop. The point is falsification: if the high
    bucket's forward drawdowns are no worse than the low bucket's, the gauge is
    noise; if they are worse but the gauge fires constantly, it is still mostly
    false alarms. Both facts come out of the returned report.

    ``scores`` may be passed to reuse a precomputed :meth:`TailRiskGauge.series`.
    """
    closes = np.asarray(closes, dtype=float)
    gauge = gauge or TailRiskGauge()
    if scores is None:
        scores = gauge.series(closes)
    scores = np.asarray(scores, dtype=float)

    fdd = forward_drawdown(closes, horizon)
    valid = np.isfinite(fdd) & np.isfinite(scores)
    s = scores[valid]
    d = fdd[valid]
    n = int(s.size)

    high = s >= threshold
    low = ~high
    dh, dl = d[high], d[low]
    n_high, n_low = int(dh.size), int(dl.size)

    def _mean(a: np.ndarray) -> float:
        return float(a.mean()) if a.size else 0.0

    def _median(a: np.ndarray) -> float:
        return float(np.median(a)) if a.size else 0.0

    def _worst(a: np.ndarray) -> float:
        return float(a.min()) if a.size else 0.0

    breach = d <= -abs(tail_move)
    base_rate = float(breach.mean()) if n else 0.0
    hit_rate_high = float((dh <= -abs(tail_move)).mean()) if n_high else 0.0

    return FalseAlarmReport(
        threshold=threshold, horizon=horizon, n=n,
        alarm_rate=float(n_high / n) if n else 0.0,
        n_high=n_high, n_low=n_low,
        mean_fdd_high=_mean(dh), mean_fdd_low=_mean(dl),
        median_fdd_high=_median(dh), median_fdd_low=_median(dl),
        worst_fdd_high=_worst(dh), worst_fdd_low=_worst(dl),
        tail_move=abs(tail_move),
        hit_rate_high=hit_rate_high,
        false_alarm_rate_high=1.0 - hit_rate_high,
        base_rate=base_rate,
    )


__all__ = [
    "COMPONENTS", "EVTFit", "FalseAlarmReport", "TailReading", "TailRiskGauge",
    "combine_regime_risk", "evaluate_false_alarms", "forward_drawdown",
]
