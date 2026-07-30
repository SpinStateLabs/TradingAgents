"""The decision loop and the self-improvement machinery around it.

The two-tier loop (fast quant execution under a slow, expiring LLM mandate)
lives in :mod:`spintrader.loop.decision_loop`. The pieces it composes:

* :mod:`spintrader.loop.context`  -- the causal market snapshot both tiers read
* :mod:`spintrader.loop.voting`   -- personas -> votes (LLM or a quant fallback)
* :mod:`spintrader.loop.mandate`  -- votes -> panel verdict -> Mandate
"""

from spintrader.loop.context import MarketContext, build_context, gather_contexts
from spintrader.loop.decision_loop import (
    DecisionLoop, ExecutionResult, LiveCursor, build_paper_loop,
)
from spintrader.loop.mandate import Deliberation, MandateService, build_mandate
from spintrader.loop.voting import BootstrapVoter, LLMVoter, VoteItem, Voter

__all__ = [
    "BootstrapVoter", "DecisionLoop", "Deliberation", "ExecutionResult",
    "LLMVoter", "LiveCursor", "MandateService", "MarketContext", "VoteItem",
    "Voter", "build_context", "build_mandate", "build_paper_loop",
    "gather_contexts",
]
