"""Candidate generation for the improvement cycle -- the agent factory.

The self-improvement loop needs a supply of *candidates* to evaluate. This
module produces them as concrete configurations of strategy **families**: a
family is a strategy class plus a base config and a parameter grid. v1 ships two
families that forecast expected return in opposite ways --
:class:`~spintrader.agents.personas.baseline_trend.BaselineTrendAgent` (buy
strength, expect continuation) and
:class:`~spintrader.agents.personas.mean_reversion.MeanReversionAgent` (buy
weakness, expect reversion). Searching across families, not just parameters, is
how the loop attacks the *edge* rather than the trading rule -- which the
baseline decomposition identified as the only lever that matters.

Two properties matter for a search loop and are both enforced here:

* **Determinism.** Candidates are enumerated from each family's grid in a fixed
  order, so a round is reproducible and a promotion can be re-derived. The grid
  also bounds the search -- the loop cannot quietly test more configurations than
  the grids contain, which keeps the multiple-testing accounting honest.
* **Validity.** Combinations a strategy's constructor would reject are filtered
  before they are ever counted as a trial, so a malformed config never inflates
  the trial ledger.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from spintrader.agents.personas.baseline_trend import BaselineTrendAgent
from spintrader.agents.personas.hedge import HedgeEnsembleAgent
from spintrader.agents.personas.markov_chain import HighOrderMarkovAgent
from spintrader.agents.personas.mean_reversion import MeanReversionAgent
from spintrader.agents.personas.regime_switch import RegimeSwitchingAgent
from spintrader.agents.personas.sentiment import SentimentAgent

if TYPE_CHECKING:
    from spintrader.core.types import Bar
    from spintrader.data.sentiment import SentimentFeed

# A per-fold context builder: given a window of bars it returns the extra
# constructor kwargs a strategy needs but bars do not carry (a sentiment map,
# say). ``run_backtest``/``walk_forward`` call it with the fold's own window.
ContextProvider = Callable[[Sequence["Bar"]], "dict[str, Any]"]

# --- trend family -----------------------------------------------------------

TREND_BASE: dict[str, Any] = {
    "fast_window": 20, "slow_window": 100, "vol_window": 20,
    "vol_ceiling": "0.40", "stop_pct": "0.05", "trail_pct": "0.08",
}
TREND_GRID: dict[str, Sequence[Any]] = {
    "fast_window": (10, 20, 30),
    "slow_window": (60, 100, 150),
    "vol_window": (14, 20),
    "trail_pct": ("0.05", "0.08", "0.12"),
}

# Kept as the module-level names the earlier single-family API exposed.
BASE_CONFIG = TREND_BASE
DEFAULT_GRID = TREND_GRID

# --- mean-reversion family --------------------------------------------------

MEANREV_BASE: dict[str, Any] = {
    "lookback": 50, "entry_z": "1.5", "exit_z": "0.0",
    "vol_ceiling": "0.40", "stop_pct": "0.05", "trail_pct": "0.08",
}
MEANREV_GRID: dict[str, Sequence[Any]] = {
    "lookback": (30, 60, 120),
    "entry_z": ("1.0", "1.5", "2.0"),
    "trail_pct": ("0.05", "0.08"),
}

# --- high-order Markov chain family -----------------------------------------

MARKOV_BASE: dict[str, Any] = {
    "order": 1, "n_states": 3, "lookback": 200, "dead_zone": "0.25",
    "stop_pct": "0.05", "trail_pct": "0.08",
}
MARKOV_GRID: dict[str, Sequence[Any]] = {
    "order": (1, 2, 3),
    "n_states": (2, 3),
    "lookback": (200, 400),
}

# --- Markov regime-switching family (HMM; needs hmmlearn to actually fit) ----

REGIME_BASE: dict[str, Any] = {
    "n_states": 3, "fit_window": 500, "refit_interval": 250,
    "risk_on": "0.40", "risk_off": "0.60", "drift_window": 30,
}
REGIME_GRID: dict[str, Sequence[Any]] = {
    "n_states": (2, 3),
    "risk_on": ("0.34", "0.40"),
}

# --- game-theory (no-regret / Hedge) family ---------------------------------

HEDGE_BASE: dict[str, Any] = {
    "lookback": 200, "eta": "2.0", "entry_threshold": "0.15",
    "fast_window": 10, "slow_window": 30,
}
HEDGE_GRID: dict[str, Sequence[Any]] = {
    "eta": ("1.0", "2.0", "4.0"),
    "entry_threshold": ("0.1", "0.2"),
}


def _trend_valid(params: Mapping[str, Any]) -> bool:
    fast = int(params.get("fast_window", TREND_BASE["fast_window"]))
    slow = int(params.get("slow_window", TREND_BASE["slow_window"]))
    vol = int(params.get("vol_window", TREND_BASE["vol_window"]))
    return fast >= 2 and vol >= 2 and slow > fast


def _meanrev_valid(params: Mapping[str, Any]) -> bool:
    return int(params.get("lookback", MEANREV_BASE["lookback"])) >= 5


# --- sentiment family -------------------------------------------------------
#
# Unlike the price families, the sentiment persona forecasts from a channel bars
# do not carry, so its per-fold sentiment map is supplied through a
# ``context_provider`` rather than a grid axis. The grid searches only the rule
# parameters; the feed is threaded in per fold (see :func:`sentiment_family`).

SENTIMENT_BASE: dict[str, Any] = {
    "vol_lookback": 20, "entry_threshold": "0.35", "exit_threshold": "0.10",
    "vol_ceiling": "2.0", "stop_pct": "0.05", "trail_pct": "0.08",
    "min_mentions": 0,
}
SENTIMENT_GRID: dict[str, Sequence[Any]] = {
    "entry_threshold": ("0.30", "0.35", "0.45"),
    "exit_threshold": ("0.05", "0.10"),
    "vol_lookback": (14, 20),
}


def _sentiment_valid(params: Mapping[str, Any]) -> bool:
    # Mirror the persona's own constructor guards so a config it would reject is
    # never counted as a trial: a hysteresis band (exit below entry) and enough
    # lookback to estimate volatility.
    entry = float(params.get("entry_threshold", SENTIMENT_BASE["entry_threshold"]))
    exit_ = float(params.get("exit_threshold", SENTIMENT_BASE["exit_threshold"]))
    lookback = int(params.get("vol_lookback", SENTIMENT_BASE["vol_lookback"]))
    return lookback >= 5 and -1.0 < entry <= 1.0 and exit_ < entry


@dataclass(slots=True)
class StrategyFamily:
    """A strategy class plus the base config and grid to search over it.

    ``context_provider`` is optional and, when set, is a *builder*: given the
    run's symbol it returns a per-fold :data:`ContextProvider`. It exists for
    families like sentiment whose strategy needs a per-bar series that bars do
    not carry; the builder closes over the data source (a feed) and the per-fold
    callable it returns is what slices that source to each fold's window. Price
    families leave it ``None`` and are constructed from grid params alone.
    """
    key: str
    strategy_cls: type
    base: dict[str, Any]
    grid: dict[str, Sequence[Any]]
    valid: Callable[[Mapping[str, Any]], bool] = lambda _p: True
    context_provider: Callable[[str], ContextProvider] | None = None


def trend_family() -> StrategyFamily:
    return StrategyFamily("trend", BaselineTrendAgent, dict(TREND_BASE),
                          {k: tuple(v) for k, v in TREND_GRID.items()}, _trend_valid)


def mean_reversion_family() -> StrategyFamily:
    return StrategyFamily("mean_reversion", MeanReversionAgent, dict(MEANREV_BASE),
                          {k: tuple(v) for k, v in MEANREV_GRID.items()}, _meanrev_valid)


def markov_family() -> StrategyFamily:
    return StrategyFamily("markov_chain", HighOrderMarkovAgent, dict(MARKOV_BASE),
                          {k: tuple(v) for k, v in MARKOV_GRID.items()})


def regime_switching_family() -> StrategyFamily:
    return StrategyFamily("regime_switch", RegimeSwitchingAgent, dict(REGIME_BASE),
                          {k: tuple(v) for k, v in REGIME_GRID.items()})


def hedge_family() -> StrategyFamily:
    return StrategyFamily("hedge", HedgeEnsembleAgent, dict(HEDGE_BASE),
                          {k: tuple(v) for k, v in HEDGE_GRID.items()})


def sentiment_family(
    feed: "SentimentFeed", interval_minutes: int = 1440,
) -> StrategyFamily:
    """A searchable :class:`~spintrader.agents.personas.sentiment.SentimentAgent`
    family that carries a live sentiment ``feed``.

    This is the seam that lets sentiment be searched like any other family. The
    grid varies only the persona's rule parameters; the sentiment map itself is
    injected per fold by the returned family's ``context_provider``. That
    provider re-fetches and re-aligns the feed for **each fold's own window**,
    which is what keeps a walk-forward causal here: a fold's sentiment is sliced
    to ``[window_start, window_end]``, so it can never contain a score aligned to
    a later fold's bar. The slice is inclusive of the window's own close-times
    and excludes everything after, and the feed is fetched from one interval
    before the window so the first bar's bucket is complete without reaching
    forward.

    ``interval_minutes`` is the sentiment bucketing cadence and should match the
    bar interval (a daily backtest buckets sentiment daily -- 1440 minutes).
    """
    delta = timedelta(minutes=interval_minutes)

    def provider_for(symbol: str) -> ContextProvider:
        def provide(window: "Sequence[Bar]") -> dict[str, Any]:
            if not window:
                return {"sentiment": {}}
            start, end = window[0].ts, window[-1].ts
            mapping = feed.mapping(
                symbol, since=start - delta, interval_minutes=interval_minutes,
            )
            # Scope to this fold's window: past enough to seed the first bar,
            # never past its end. This inclusive slice is the causal guarantee.
            scoped = {ts: sc for ts, sc in mapping.items() if start <= ts <= end}
            return {"sentiment": scoped}

        return provide

    return StrategyFamily(
        "sentiment", SentimentAgent, dict(SENTIMENT_BASE),
        {k: tuple(v) for k, v in SENTIMENT_GRID.items()},
        _sentiment_valid, context_provider=provider_for,
    )


def default_families() -> list[StrategyFamily]:
    """Every shipped family, forecasting expected return a different way.

    trend / mean-reversion (price level), high-order Markov chain (return-symbol
    transitions), Markov regime-switching (latent HMM state), and a game-theory
    no-regret ensemble. The regime family needs ``hmmlearn`` to fit and otherwise
    stands aside; the rest are pure and run anywhere.
    """
    return [trend_family(), mean_reversion_family(), markov_family(),
            regime_switching_family(), hedge_family()]


@dataclass(slots=True)
class CandidateConfig:
    """One concrete strategy configuration to evaluate.

    ``key`` is a stable content hash over the family *and* the parameters, so a
    trend and a mean-reversion config that happen to share a value are distinct,
    and the research memory can recognise a configuration it has already tested.
    """
    params: dict[str, Any]
    family: str = "trend"
    strategy_cls: type | None = None

    @property
    def key(self) -> str:
        blob = json.dumps(
            {"_family": self.family,
             **{k: str(v) for k, v in sorted(self.params.items())}},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"{self.family}_{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def describe(self) -> str:
        varied = ", ".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.name}({varied})"


class CandidateFactory:
    """Enumerates strategy configurations across one or more families.

    Backward-compatible: ``CandidateFactory()`` searches the trend family, and
    ``CandidateFactory(grid=...)`` overrides its grid, exactly as the earlier
    single-family API did. Pass ``families=[...]`` to search several.
    """

    def __init__(
        self,
        base: Mapping[str, Any] | None = None,
        grid: Mapping[str, Sequence[Any]] | None = None,
        families: Sequence[StrategyFamily] | None = None,
    ) -> None:
        if families is not None:
            self.families = list(families)
        else:
            fam = trend_family()
            if base is not None:
                fam.base = dict(base)
            if grid is not None:
                fam.grid = {k: tuple(v) for k, v in grid.items()}
            self.families = [fam]

    def all_configs(self) -> list[CandidateConfig]:
        """Every valid configuration across all families, in a fixed order."""
        out: list[CandidateConfig] = []
        seen: set[str] = set()
        for family in self.families:
            axes = sorted(family.grid)
            for combo in itertools.product(*(family.grid[a] for a in axes)):
                params = dict(family.base)
                params.update(dict(zip(axes, combo)))
                if not family.valid(params):
                    continue
                cfg = CandidateConfig(params, family=family.key,
                                      strategy_cls=family.strategy_cls)
                if cfg.key in seen:
                    continue
                seen.add(cfg.key)
                out.append(cfg)
        return out

    def generate(
        self, n: int | None = None, avoid: Iterable[str] = (),
    ) -> list[CandidateConfig]:
        """Return up to ``n`` configs whose keys are not in ``avoid``."""
        avoid = set(avoid)
        fresh = [c for c in self.all_configs() if c.key not in avoid]
        return fresh if n is None else fresh[:n]

    def family_classes(self) -> dict[str, type]:
        """Map each family key to its strategy class (to re-backtest a champion)."""
        return {f.key: f.strategy_cls for f in self.families}

    def __len__(self) -> int:
        return len(self.all_configs())


__all__ = [
    "BASE_CONFIG", "DEFAULT_GRID", "HEDGE_BASE", "HEDGE_GRID", "MARKOV_BASE",
    "MARKOV_GRID", "MEANREV_BASE", "MEANREV_GRID", "REGIME_BASE", "REGIME_GRID",
    "SENTIMENT_BASE", "SENTIMENT_GRID", "TREND_BASE", "TREND_GRID",
    "CandidateConfig", "CandidateFactory", "ContextProvider", "StrategyFamily",
    "default_families", "hedge_family", "markov_family", "mean_reversion_family",
    "regime_switching_family", "sentiment_family", "trend_family",
]
