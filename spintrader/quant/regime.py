"""Hidden Markov regime detection.

Markets are not one process. Returns during a calm uptrend and returns during
a liquidation cascade have different means, different variances and different
persistence, and a single model fitted across both describes neither. An HMM
makes that explicit: latent states with their own emission distributions, and
a transition matrix encoding how sticky each state is.

Three failure modes dominate, and each is handled explicitly below.

**1. Smoothed states are lookahead.**
``hmmlearn.predict`` runs Viterbi over the *entire* sequence, so the state it
assigns to day 100 depends on data from day 500. Using that as a historical
trading signal is not subtle overfitting -- it is reading the future. The
resulting backtest is spectacular and entirely fake. :meth:`RegimeModel.filter`
therefore runs the forward algorithm only, so the state at ``t`` is
conditioned on data up to ``t`` alone. Smoothed states remain available under
a name that says what they are, for research plots and nothing else.

**2. Label switching.**
EM assigns state indices arbitrarily. Refit on one more day of data and
yesterday's "state 0" may become today's "state 2", so a strategy keyed on
state index silently inverts. States are therefore canonically ordered by
volatility after fitting, making index meaningful and stable across refits.

**3. Local optima.**
EM converges to whatever basin it starts in, and a bad basin yields states
that split on nothing. Multiple random restarts are run and the best
likelihood kept.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from spintrader.quant.features import FeatureMatrix

log = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


class RegimeError(RuntimeError):
    """Regime model fitting or inference failed."""


# Canonical labels, ordered from calmest to most stressed. The ordering is the
# contract: state 0 is always the quietest regime.
REGIME_LABELS_3 = ("calm", "normal", "stressed")
REGIME_LABELS_4 = ("calm", "normal", "volatile", "crisis")
REGIME_LABELS_2 = ("calm", "stressed")


def labels_for(n_states: int) -> tuple[str, ...]:
    return {
        2: REGIME_LABELS_2, 3: REGIME_LABELS_3, 4: REGIME_LABELS_4,
    }.get(n_states, tuple(f"state_{i}" for i in range(n_states)))


@dataclass(slots=True)
class RegimeState:
    """The regime inferred for a single point in time."""
    ts: datetime
    state: int
    label: str
    probabilities: np.ndarray      # posterior over states, sums to 1
    risk_score: float              # 0 benign .. 1 crisis

    @property
    def confidence(self) -> float:
        """Posterior mass on the most likely state.

        A 0.51 confidence means the model is nearly indifferent, and sizing as
        though the regime were known is exactly how a risk model becomes a
        source of risk.
        """
        return float(self.probabilities.max())

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": int(self.state),
            "label": self.label,
            "probabilities": [float(p) for p in self.probabilities],
            "risk_score": float(self.risk_score),
            "confidence": self.confidence,
        }


@dataclass
class RegimeModel:
    """Gaussian HMM over causal market features."""

    n_states: int = 3
    covariance_type: str = "full"
    n_restarts: int = 8
    max_iter: int = 200
    random_state: int = 17
    feature_names: tuple[str, ...] = ()
    # Populated by fit()
    _hmm: Any = field(default=None, repr=False)
    _order: np.ndarray | None = field(default=None, repr=False)
    _vol_index: int = field(default=1, repr=False)
    _mean_vol: np.ndarray | None = field(default=None, repr=False)
    _standardised_vol: np.ndarray | None = field(default=None, repr=False)
    _scaler: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    fit_start: datetime | None = None
    fit_end: datetime | None = None
    log_likelihood: float | None = None
    converged: bool = False

    # -- fitting -----------------------------------------------------------

    def fit(self, features: FeatureMatrix) -> "RegimeModel":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise RegimeError("hmmlearn is not installed") from exc

        X = self._standardise(features.values)
        if X.shape[0] < self.n_states * 10:
            raise RegimeError(
                f"need at least {self.n_states * 10} observations to fit "
                f"{self.n_states} states, got {X.shape[0]}"
            )

        best, best_score = None, -np.inf
        for restart in range(self.n_restarts):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.max_iter,
                random_state=self.random_state + restart,
                init_params="stmc",
            )
            with warnings.catch_warnings():
                # hmmlearn warns loudly on non-convergence; we handle it via
                # the restart loop and report it on the model instead.
                warnings.simplefilter("ignore")
                try:
                    model.fit(X)
                    score = model.score(X)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    log.debug("regime restart %d failed: %s", restart, exc)
                    continue
            if np.isfinite(score) and score > best_score:
                best, best_score = model, score

        if best is None:
            raise RegimeError(
                f"all {self.n_restarts} EM restarts failed to fit a "
                f"{self.n_states}-state model"
            )

        self._hmm = best
        self.log_likelihood = float(best_score)
        self.converged = bool(getattr(best.monitor_, "converged", False))
        self.feature_names = features.names
        self.fit_start = features.timestamps[0]
        self.fit_end = features.timestamps[-1]
        self._canonicalise()

        if not self.converged:
            log.warning(
                "regime model did not converge in %d iterations; states may be "
                "unstable", self.max_iter,
            )
        return self

    def _standardise(self, values: np.ndarray) -> np.ndarray:
        """Z-score features using fit-set statistics only.

        Stored on first call so that inference standardises with the same
        constants -- rescaling inference data by its own statistics would leak
        information from the inference window back into the features.
        """
        if self._scaler is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std < 1e-12] = 1.0
            self._scaler = (mean, std)
        mean, std = self._scaler
        return (values - mean) / std

    def _canonicalise(self) -> None:
        """Order states by mean volatility, calmest first.

        Without this, state indices are arbitrary per fit and any strategy
        keyed on them silently inverts when the model is refitted.
        """
        try:
            vol_index = self.feature_names.index("vol")
        except ValueError:
            vol_index = 1
        self._vol_index = vol_index

        # means_ are in STANDARDISED units, because fit() z-scores the
        # features. Ordering on them is valid -- standardisation is monotonic
        # -- but reporting them as volatility is not: a z-score is routinely
        # negative, and "annualised volatility -90%" is nonsense.
        standardised_means = self._hmm.means_[:, vol_index]
        order = np.argsort(standardised_means)
        self._order = order
        self._standardised_vol = standardised_means[order]

        # Invert the standardisation to recover interpretable volatilities.
        mean, std = self._scaler                        # type: ignore[attr-defined]
        self._mean_vol = standardised_means[order] * std[vol_index] + mean[vol_index]

    def _remap(self, values: np.ndarray, axis: int = -1) -> np.ndarray:
        """Reorder a per-state array into canonical order."""
        if self._order is None:
            return values
        return np.take(values, self._order, axis=axis)

    # -- inference ---------------------------------------------------------

    def filter(self, features: FeatureMatrix) -> list[RegimeState]:
        """Causal state inference: the state at t uses data up to t only.

        This is the ONLY inference method whose output may drive a trading
        decision. It runs the forward algorithm and reads the filtered
        posterior at each step.
        """
        self._require_fit()
        X = self._standardise(features.values)

        # hmmlearn exposes the forward pass through _compute_log_likelihood
        # plus its internal forward routine; running it explicitly keeps the
        # causality guarantee visible rather than trusting a wrapper.
        framelogprob = self._hmm._compute_log_likelihood(X)
        log_startprob = np.log(np.maximum(self._hmm.startprob_, 1e-300))
        log_transmat = np.log(np.maximum(self._hmm.transmat_, 1e-300))

        n_samples, n_states = framelogprob.shape
        log_alpha = np.zeros((n_samples, n_states))
        log_alpha[0] = log_startprob + framelogprob[0]
        for t in range(1, n_samples):
            for j in range(n_states):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + log_transmat[:, j])
                    + framelogprob[t, j]
                )

        states: list[RegimeState] = []
        for t in range(n_samples):
            posterior = _softmax(log_alpha[t])
            posterior = self._remap(posterior)
            index = int(np.argmax(posterior))
            states.append(RegimeState(
                ts=features.timestamps[t],
                state=index,
                label=labels_for(self.n_states)[index],
                probabilities=posterior,
                risk_score=self.risk_score(posterior),
            ))
        return states

    def filter_latest(self, features: FeatureMatrix) -> RegimeState:
        """The current regime. Convenience wrapper over :meth:`filter`."""
        states = self.filter(features)
        if not states:
            raise RegimeError("no observations to infer a regime from")
        return states[-1]

    def smoothed_states_research_only(self, features: FeatureMatrix) -> np.ndarray:
        """Viterbi over the whole sequence.

        USES FUTURE DATA. The state assigned to an early observation depends
        on later ones, so this must never touch a trading decision or a
        backtest signal. It exists for research plots, where seeing the
        model's best retrospective segmentation is genuinely useful.
        """
        self._require_fit()
        raw = self._hmm.predict(self._standardise(features.values))
        # Map raw indices into canonical order.
        inverse = np.argsort(self._order) if self._order is not None else np.arange(self.n_states)
        return inverse[raw]

    def risk_score(self, posterior: np.ndarray) -> float:
        """Map a state posterior to a 0..1 risk score.

        Probability-weighted rather than argmax: a 55/45 split between calm
        and crisis should produce a middling score, not a confident 'calm'.
        This is what feeds RiskProfile.scaled_for_regime, so an overconfident
        score here becomes an oversized position downstream.
        """
        if self.n_states == 1:
            return 0.0
        weights = np.linspace(0.0, 1.0, self.n_states)
        return float(np.clip(np.dot(posterior, weights), 0.0, 1.0))

    # -- diagnostics -------------------------------------------------------

    def _require_fit(self) -> None:
        if self._hmm is None:
            raise RegimeError("model is not fitted; call fit() first")

    def state_volatility(self) -> np.ndarray:
        """Mean annualised volatility of each state, in canonical order.

        Inverse-transformed out of standardised space, so these are readable
        as percentages. Always non-negative for a well-fitted model; a
        negative value here means the inverse transform is broken, not that
        the market had negative variance.
        """
        self._require_fit()
        if self._mean_vol is None:
            return np.array([])
        return np.maximum(self._mean_vol, 0.0)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Transition probabilities in canonical state order."""
        self._require_fit()
        remapped = self._remap(self._hmm.transmat_, axis=0)
        return self._remap(remapped, axis=1)

    def expected_durations(self) -> np.ndarray:
        """Expected persistence of each state, in bars.

        For a Markov chain the sojourn time is geometric with mean
        1/(1-p_ii). A regime whose expected duration is one or two bars is not
        a regime -- it is noise the model has labelled, and the fit should be
        rejected.
        """
        diag = np.diag(self.transition_matrix)
        with np.errstate(divide="ignore"):
            return np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)

    def n_parameters(self) -> int:
        """Free parameters, for information criteria."""
        k, d = self.n_states, len(self.feature_names) or 1
        transitions = k * (k - 1)
        starts = k - 1
        means = k * d
        if self.covariance_type == "full":
            covars = k * d * (d + 1) // 2
        elif self.covariance_type == "diag":
            covars = k * d
        else:
            covars = k
        return transitions + starts + means + covars

    def bic(self, n_samples: int) -> float:
        """Bayesian information criterion. Lower is better.

        BIC rather than AIC because it penalises parameters more heavily, and
        an over-parameterised regime model is the more expensive error: it
        finds regimes in noise, and the strategy then trades those.
        """
        if self.log_likelihood is None:
            raise RegimeError("model is not fitted")
        return -2.0 * self.log_likelihood + self.n_parameters() * np.log(n_samples)

    def summary(self) -> dict[str, Any]:
        self._require_fit()
        durations = self.expected_durations()
        return {
            "n_states": self.n_states,
            "labels": list(labels_for(self.n_states)),
            "converged": self.converged,
            "log_likelihood": self.log_likelihood,
            "bic": self.bic(1),           # caller supplies n for a real value
            "annualised_volatility_by_state": self.state_volatility().tolist(),
            "expected_duration_bars": [float(d) for d in durations],
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "model_version": MODEL_VERSION,
        }


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------

def select_n_states(
    features: FeatureMatrix,
    candidates: Sequence[int] = (2, 3, 4),
    min_duration_bars: float = 3.0,
    **model_kwargs: Any,
) -> tuple[RegimeModel, dict[int, float]]:
    """Fit each candidate and pick the best by BIC, rejecting unstable fits.

    A model whose states last a bar or two has not found regimes; it has
    labelled noise. Such fits are excluded regardless of likelihood, because
    BIC alone will happily prefer them.
    """
    scores: dict[int, float] = {}
    best: RegimeModel | None = None
    best_bic = np.inf

    for n in candidates:
        try:
            model = RegimeModel(n_states=n, **model_kwargs).fit(features)
        except RegimeError as exc:
            log.info("regime model with %d states could not be fitted: %s", n, exc)
            continue

        bic = model.bic(len(features))
        scores[n] = bic

        shortest = float(np.min(model.expected_durations()))
        if shortest < min_duration_bars:
            log.info(
                "rejecting %d-state model: shortest expected duration %.1f bars "
                "is below the %.1f-bar floor (states are tracking noise)",
                n, shortest, min_duration_bars,
            )
            continue

        if bic < best_bic:
            best, best_bic = model, bic

    if best is None:
        raise RegimeError(
            f"no candidate in {list(candidates)} produced a stable fit; "
            f"BIC scores: {scores}"
        )
    return best, scores


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------

def _logsumexp(values: np.ndarray) -> float:
    peak = np.max(values)
    if not np.isfinite(peak):
        return float(peak)
    return float(peak + np.log(np.sum(np.exp(values - peak))))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    total = exponentiated.sum()
    return exponentiated / total if total > 0 else np.full_like(values, 1.0 / values.size)


__all__ = [
    "MODEL_VERSION", "REGIME_LABELS_2", "REGIME_LABELS_3", "REGIME_LABELS_4",
    "RegimeError", "RegimeModel", "RegimeState", "labels_for", "select_n_states",
]
