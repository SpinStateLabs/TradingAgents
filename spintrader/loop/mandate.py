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
from spintrader.agents.personas.spec import Horizon, PersonaRegistry, PersonaSpec
from spintrader.core.types import Action, utcnow
from spintrader.risk.engine import Mandate
from spintrader.loop.context import MarketContext
from spintrader.loop.voting import Adjudicator, BootstrapVoter, VoteItem, Voter

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
    #: Instrument keys whose contested first-pass verdict was re-adjudicated on
    #: the deep model this cycle (task 18). Empty when nothing escalated or the
    #: voter has no deep tier.
    escalated: frozenset[str] = field(default_factory=frozenset)

    def summary(self) -> str:
        permits = ", ".join(sorted(self.mandate.permitted)) or "nothing"
        escalated = f", escalated {len(self.escalated)}" if self.escalated else ""
        return (
            f"mandate permits {permits}; regime_risk {self.mandate.regime_risk:.2f}, "
            f"budget x{self.mandate.risk_budget_multiplier:.2f}{escalated}, "
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
        escalate: bool = True,
    ) -> None:
        self.registry = registry
        self.panel = panel or PersonaPanel(registry)
        # Default to the no-LLM voter so the service is usable offline and in
        # tests; a deployment injects an LLMVoter bound to the GB10 router.
        self.voter = voter or BootstrapVoter()
        self.ttl = ttl
        self.min_reliability = min_reliability
        # Whether a contested first-pass verdict is re-adjudicated on the deep
        # model. The dispersion / lens-diversity thresholds that decide *what*
        # counts as contested live on the panel; this only toggles the pass.
        self.escalate = escalate

    def deliberate(
        self,
        contexts: Mapping[str, MarketContext],
        now: datetime | None = None,
    ) -> Deliberation:
        """Produce a mandate from the current market contexts."""
        now = now or utcnow()

        # Build one flat work list across all instruments, so the LLM voter can
        # batch every persona call in a single tier-grouped pass. Keep each
        # instrument's applicable specs so the escalation pass can reuse them
        # without re-querying the registry.
        specs_by_key: dict[str, list[PersonaSpec]] = {}
        items: list[VoteItem] = []
        for key, ctx in contexts.items():
            specs = self.registry.applicable(
                ctx.asset_class, ctx.horizon, min_reliability=self.min_reliability,
            )
            specs_by_key[key] = specs
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

        # Adaptive tier escalation (task 18): re-adjudicate any contested verdict
        # on the deep model and replace it in place. Mutates `verdicts`.
        escalated = self._escalate(contexts, specs_by_key, verdicts)

        mandate = build_mandate(verdicts, regime_risk, ttl=self.ttl, now=now)
        deliberation = Deliberation(
            mandate=mandate, verdicts=verdicts, contexts=dict(contexts),
            escalated=frozenset(escalated),
        )
        log.info("slow loop: %s", deliberation.summary())
        return deliberation

    def _escalate(
        self,
        contexts: Mapping[str, MarketContext],
        specs_by_key: Mapping[str, Sequence[PersonaSpec]],
        verdicts: dict[str, PanelVerdict],
    ) -> set[str]:
        """Re-adjudicate contested first-pass verdicts on the deep tier.

        Collects every instrument whose quick verdict set ``escalate`` (a split
        panel or a single lens), runs ONE batched deep pass over just those --
        all deep calls in a single ``adjudicate_all`` batch, never interleaved
        with the quick pass -- and replaces those instruments' verdicts with the
        ones re-aggregated from the deep votes.

        A graceful no-op when escalation is disabled, when the voter has no deep
        tier (the bootstrap voter is not an :class:`Adjudicator`), or when
        nothing escalated. Returns the keys that were actually re-adjudicated.
        """
        if not self.escalate:
            return set()
        # Only a voter that can reach the deep model adjudicates. The bootstrap
        # voter cannot, so escalation degrades to a no-op rather than breaking.
        if not isinstance(self.voter, Adjudicator):
            return set()

        escalated_keys = [key for key, verdict in verdicts.items() if verdict.escalate]
        if not escalated_keys:
            return set()

        # One flat deep work list across ALL escalated instruments, so the deep
        # model pages in exactly once for the whole escalation set.
        deep_items: list[VoteItem] = []
        for key in escalated_keys:
            ctx = contexts[key]
            deep_items.extend(
                VoteItem(spec=spec, context=ctx) for spec in specs_by_key[key]
            )

        deep_votes_by_key = self.voter.adjudicate_all(deep_items)

        re_adjudicated: set[str] = set()
        for key in escalated_keys:
            deep_votes = deep_votes_by_key.get(key, [])
            if not deep_votes:
                # The deep pass returned nothing for this instrument; keep the
                # quick verdict rather than blanking it.
                continue
            ctx = contexts[key]
            verdicts[key] = self.panel.aggregate(
                deep_votes, ctx.asset_class, ctx.horizon,
            )
            re_adjudicated.add(key)
        return re_adjudicated


__all__ = [
    "DEFAULT_BUDGET_FLOOR", "DEFAULT_PERMIT_CONFIDENCE", "DEFAULT_TARGET_CONFIDENCE",
    "Deliberation", "MandateService", "build_mandate",
]
