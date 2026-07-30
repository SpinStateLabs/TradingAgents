"""Candidate generation for the improvement cycle -- the agent factory (v1).

The self-improvement loop needs a supply of *candidates* to evaluate. This
module produces them as concrete parameter configurations for the deterministic
:class:`~spintrader.agents.personas.baseline_trend.BaselineTrendAgent`. Parameter
variants are the honest first step: they search a bounded, enumerable space with
no fitted state, so a promotion can be attributed to a specific configuration
rather than to an opaque model. Generating genuinely new personas is a later
extension; the interface here (a stream of typed configs, each with a stable key)
does not change when that arrives.

Two properties matter for a search loop and are both enforced here:

* **Determinism.** Candidates are enumerated from a grid in a fixed order, so a
  round is reproducible and a promotion can be re-derived. A seeded RNG would
  also work, but an enumerable grid additionally bounds the search -- the loop
  cannot quietly test more configurations than the grid contains, which is what
  keeps the multiple-testing accounting honest.
* **Validity.** Combinations the agent would reject (``slow_window`` not strictly
  greater than ``fast_window``) are filtered out before they are ever counted as
  a trial, so a malformed config never inflates the trial ledger.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

# The base configuration every candidate starts from -- the tunable parameters
# only. ``interval`` and ``continuous`` are NOT here: they are properties of the
# bars being tested, not of the strategy, and the backtest runner fills them from
# the data. Keeping them out means a config's identity is its tunables alone.
BASE_CONFIG: dict[str, Any] = {
    "fast_window": 20,
    "slow_window": 100,
    "vol_window": 20,
    "vol_ceiling": "0.40",
    "stop_pct": "0.05",
    "trail_pct": "0.08",
}

# The default search grid: the axes varied and the values tried on each. Kept
# small on purpose -- every point is a lifetime trial against the deflated
# Sharpe, so a sprawling grid does not find more signal, it just raises the bar
# the winner must clear.
DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "fast_window": (10, 20, 30),
    "slow_window": (60, 100, 150),
    "vol_window": (14, 20),
    "trail_pct": ("0.05", "0.08", "0.12"),
}


@dataclass(slots=True)
class CandidateConfig:
    """One concrete strategy configuration to evaluate.

    ``key`` is a stable content hash so the research memory can recognise a
    configuration it has already paid to test, and ``name`` is a human- and
    log-friendly label derived from it.
    """
    params: dict[str, Any]

    @property
    def key(self) -> str:
        blob = json.dumps(
            {k: str(v) for k, v in sorted(self.params.items())}, sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"cand_{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def describe(self) -> str:
        varied = ", ".join(
            f"{k}={self.params[k]}" for k in sorted(self.params)
            if k not in ("interval", "continuous")
        )
        return f"{self.name}({varied})"


def _valid(params: Mapping[str, Any]) -> bool:
    """Reject combinations the agent's own constructor would reject.

    Doing it here means an invalid point is never generated and therefore never
    counted as a trial -- the grid's effective size is the number of *valid*
    points, which is what the trial ledger should reflect.
    """
    fast = int(params.get("fast_window", BASE_CONFIG["fast_window"]))
    slow = int(params.get("slow_window", BASE_CONFIG["slow_window"]))
    vol = int(params.get("vol_window", BASE_CONFIG["vol_window"]))
    return fast >= 2 and vol >= 2 and slow > fast


class CandidateFactory:
    """Enumerates strategy configurations from a base config and a grid."""

    def __init__(
        self,
        base: Mapping[str, Any] | None = None,
        grid: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        self.base = dict(base or BASE_CONFIG)
        self.grid = {k: tuple(v) for k, v in (grid or DEFAULT_GRID).items()}

    def all_configs(self) -> list[CandidateConfig]:
        """Every valid configuration in the grid, in a fixed order."""
        axes = sorted(self.grid)
        out: list[CandidateConfig] = []
        seen: set[str] = set()
        for combo in itertools.product(*(self.grid[a] for a in axes)):
            params = dict(self.base)
            params.update(dict(zip(axes, combo)))
            if not _valid(params):
                continue
            cfg = CandidateConfig(params)
            if cfg.key in seen:          # different combos can collapse to one config
                continue
            seen.add(cfg.key)
            out.append(cfg)
        return out

    def generate(
        self, n: int | None = None, avoid: Iterable[str] = (),
    ) -> list[CandidateConfig]:
        """Return up to ``n`` configs whose keys are not in ``avoid``.

        ``avoid`` is the set of keys the research memory has already evaluated,
        so a round never re-tests -- and never re-counts -- a configuration.
        ``n=None`` returns the whole remaining grid.
        """
        avoid = set(avoid)
        fresh = [c for c in self.all_configs() if c.key not in avoid]
        return fresh if n is None else fresh[:n]

    def __len__(self) -> int:
        return len(self.all_configs())


__all__ = [
    "BASE_CONFIG", "DEFAULT_GRID", "CandidateConfig", "CandidateFactory",
]
