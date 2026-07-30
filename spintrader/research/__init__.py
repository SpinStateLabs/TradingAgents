"""Research: the generative half of the self-improvement loop.

* :mod:`spintrader.research.factory` -- candidate generation (the agent factory)
* :mod:`spintrader.research.memory`  -- what has been tried and learned

The orchestrator that drives them against the scoring half lives in
:mod:`spintrader.loop.improvement`.
"""

from spintrader.research.factory import (
    BASE_CONFIG, DEFAULT_GRID, CandidateConfig, CandidateFactory, StrategyFamily,
    default_families, mean_reversion_family, trend_family,
)
from spintrader.research.memory import ResearchMemory, TrialRecord

__all__ = [
    "BASE_CONFIG", "DEFAULT_GRID", "CandidateConfig", "CandidateFactory",
    "ResearchMemory", "StrategyFamily", "TrialRecord", "default_families",
    "mean_reversion_family", "trend_family",
]
