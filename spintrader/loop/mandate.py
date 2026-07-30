"""The slow loop: deliberate hourly, emit a Mandate the fast loop must obey.

The mandate is the one channel by which slow, expensive reasoning constrains
fast, cheap execution. It says which instruments may be *opened*, in which
direction, and how much of the risk budget a macro view justifies -- and it
expires, so a view formed under conditions that no longer hold cannot drive
minute execution indefinitely.

Three conventions are load-bearing and are asserted by tests elsewhere:

* **Empty permits nothing.** A cycle that produced no confident directional
  verdict yields an empty ``permitted`` set, which forbids all new positions.
  Defaulting the other way would let a failed deliberation authorise the whole
  universe. (The fast loop can still *exit* held positions -- that path does not
  consult the mandate's permit set -- but it cannot open.)
* **Regime risk is portfolio-wide and pessimistic.** The ``Mandate`` carries a
  single regime scalar that scales exposure for every instrument, so the most
  stressed regime in the traded set sets the cap. One calm name must not buy
  back the exposure a crisis elsewhere just removed.
* **Abstention and HOLD do not permit.** Only an actual BUY or SELL verdict,
  above a confidence floor, puts an instrument in play.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping, Sequence

from spintrader.agents.panel import PanelVerdict, PersonaPanel
from spintrader.agents.personas.spec import Horizon, PersonaRegistry
from spintrader.core.types import Action, utcnow
from spintrader.risk.engine import Mandate
from spintrader.loop.context import MarketContext
from spintrader.loop.voting import BootstrapVoter, VoteItem, Voter

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")

# Below this panel confidence an instrument is not put in play. Deliberately
# low: the risk engine applies the real per-trade confidence floor to each
# intent, so this only screens out instruments the panel has essentially no
# directional view on.
DEFAULT_PERMIT_CONFIDENCE = Decimal("0.15")

# Panel confidence at or above which the full per-position budget is granted;
# below it the budget is scaled down, never below the floor.
DEFAULT_TARGET_CONFIDENCE = Decimal("0.60")
DEFAULT_BUDGET_FLOOR = Decimal("0.40")


def build_mandate(
    verdicts: Mapping[str, PanelVerdict],
    regime_risk: Mapping[str, Decimal],
    *,
    ttl: timedelta,
    now: datetime | None = None,
    permit_confidence: Decimal = DEFAULT_PERMIT_CONFIDENCE,
    target_confidence: Decimal = DEFAULT_TARGET_CONFIDENCE,
    budget_floor: Decimal = DEFAULT_BUDGET_FLOOR,
) -> Mandate:
    """Assemble a :class:`Mandate` from per-instrument panel verdicts."""
    now = now or utcnow()

    permitted: set[str] = set()
    bias: dict[str, Decimal] = {}
    thesis: dict[str, str] = {}
    confidences: list[Decimal] = []

    for key, verdict in verdicts.items():
        if verdict.action in (Action.BUY, Action.SELL) and verdict.confidence >= permit_confidence:
            permitted.add(key)
            bias[key] = verdict.net_direction
            thesis[key] = verdict.summary()
            confidences.append(verdict.confidence)

    # Portfolio-wide regime risk: the most stressed name sets the cap.
    portfolio_regime = max(regime_risk.values(), default=ZERO)

    # Macro conviction scales the per-position budget, on top of (not instead of)
    # each intent's own confidence in the risk engine.
    if confidences:
        mean_conf = sum(confidences, ZERO) / Decimal(len(confidences))
        multiplier = min(ONE, mean_conf / target_confidence) if target_confidence > ZERO else ONE
        multiplier = max(budget_floor, multiplier)
    else:
        multiplier = ONE            # irrelevant: nothing is permitted

    return Mandate(
        issued_at=now,
        expires_at=now + ttl,
        permitted=frozenset(permitted),
        directional_bias=bias,
        risk_budget_multiplier=multiplier,
        thesis=thesis,
        regime_risk=portfolio_regime,
    )


@dataclass(slots=True)
class Deliberation:
    """One slow-cycle output: the mandate plus the reasoning behind it.

    The verdicts and contexts travel with the mandate so a decision can be
    explained later -- which is the whole point of keeping per-persona
    attribution rather than a single opaque score.
    """
    mandate: Mandate
    verdicts: dict[str, PanelVerdict] = field(default_factory=dict)
    contexts: dict[str, MarketContext] = field(default_factory=dict)

    def summary(self) -> str:
        permits = ", ".join(sorted(self.mandate.permitted)) or "nothing"
        return (
            f"mandate permits {permits}; regime_risk {self.mandate.regime_risk:.2f}, "
            f"budget x{self.mandate.risk_budget_multiplier:.2f}, "
            f"expires {self.mandate.expires_at:%Y-%m-%d %H:%M}"
        )


class MandateService:
    """Runs a deliberation cycle: contexts -> votes -> verdicts -> mandate."""

    def __init__(
        self,
        registry: PersonaRegistry,
        panel: PersonaPanel | None = None,
        voter: Voter | None = None,
        ttl: timedelta = timedelta(minutes=90),
        min_reliability: Decimal = ZERO,
    ) -> None:
        self.registry = registry
        self.panel = panel or PersonaPanel(registry)
        # Default to the no-LLM voter so the service is usable offline and in
        # tests; a deployment injects an LLMVoter bound to the GB10 router.
        self.voter = voter or BootstrapVoter()
        self.ttl = ttl
        self.min_reliability = min_reliability

    def deliberate(
        self,
        contexts: Mapping[str, MarketContext],
        now: datetime | None = None,
    ) -> Deliberation:
        """Produce a mandate from the current market contexts."""
        now = now or utcnow()

        # Build one flat work list across all instruments, so the LLM voter can
        # batch every persona call in a single tier-grouped pass.
        items: list[VoteItem] = []
        for key, ctx in contexts.items():
            specs = self.registry.applicable(
                ctx.asset_class, ctx.horizon, min_reliability=self.min_reliability,
            )
            items.extend(VoteItem(spec=spec, context=ctx) for spec in specs)

        votes_by_key = self.voter.vote_all(items)

        verdicts: dict[str, PanelVerdict] = {}
        regime_risk: dict[str, Decimal] = {}
        for key, ctx in contexts.items():
            verdict = self.panel.aggregate(
                votes_by_key.get(key, []), ctx.asset_class, ctx.horizon,
            )
            verdicts[key] = verdict
            regime_risk[key] = ctx.regime_risk

        mandate = build_mandate(verdicts, regime_risk, ttl=self.ttl, now=now)
        deliberation = Deliberation(
            mandate=mandate, verdicts=verdicts, contexts=dict(contexts),
        )
        log.info("slow loop: %s", deliberation.summary())
        return deliberation


__all__ = [
    "DEFAULT_BUDGET_FLOOR", "DEFAULT_PERMIT_CONFIDENCE", "DEFAULT_TARGET_CONFIDENCE",
    "Deliberation", "MandateService", "build_mandate",
]
