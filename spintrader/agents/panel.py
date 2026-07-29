"""The persona panel: collect votes, aggregate, and measure disagreement.

Aggregation choices here are deliberate and several are counterintuitive.

**Abstention is not a neutral vote.** A persona that cannot speak to a decision
is excluded from the denominator entirely. Counting it as neutral would let a
roster of mostly-inapplicable personas dilute the few informed ones toward
inaction -- on an hourly crypto decision that would silence Simons and Thorp
under the weight of Buffett and Burry saying nothing.

**Disagreement is signal, not noise.** When personas reasoning from *different*
evidence reach the same conclusion, that is corroboration. When they diverge,
the decision is genuinely uncertain and should be sized down and escalated to
the deep model. :attr:`PanelVerdict.dispersion` drives both.

**Lens concentration is penalised.** Five personas agreeing because they all
read momentum is one opinion, not five. Confidence is discounted when the
agreeing votes share a single lens.

**Weights are earned.** Each vote is weighted by the persona's reliability,
which the self-improvement loop sets from realised attribution, and by how well
the decision's horizon fits its native one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from spintrader.agents.personas.spec import (
    Horizon, Lens, PersonaRegistry, PersonaSpec,
)
from spintrader.core.types import Action, AssetClass, Decision, to_decimal, utcnow

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class PersonaVote:
    """One persona's opinion on one decision."""
    persona_key: str
    action: Action
    confidence: Decimal              # 0..1, 0 means abstain
    rationale: str
    changed_by: str = ""             # the evidence that would flip this view
    lenses: tuple[Lens, ...] = ()
    weight: Decimal = ONE            # reliability x horizon fit, set by the panel

    @property
    def abstained(self) -> bool:
        return self.confidence <= ZERO or self.action is Action.HOLD

    @property
    def direction(self) -> Decimal:
        """Signed conviction: +1 fully long, -1 fully short, 0 neutral."""
        if self.action is Action.BUY:
            return self.confidence
        if self.action in (Action.SELL, Action.CLOSE):
            return -self.confidence
        return ZERO

    @property
    def effective(self) -> Decimal:
        return self.direction * self.weight


@dataclass(slots=True)
class PanelVerdict:
    """Aggregated panel output, with the diagnostics needed to trust it."""
    action: Action
    confidence: Decimal
    net_direction: Decimal
    votes: list[PersonaVote] = field(default_factory=list)
    abstentions: dict[str, str] = field(default_factory=dict)
    dispersion: Decimal = ZERO           # 0 unanimous .. 1 maximally split
    lens_diversity: int = 0              # distinct lenses among agreeing votes
    participating: int = 0
    escalate: bool = False               # hand to the deep model to adjudicate

    @property
    def voted(self) -> list[PersonaVote]:
        return [v for v in self.votes if not v.abstained]

    def contributions(self) -> dict[str, object]:
        """Per-persona record for the decision audit trail.

        This is what makes attribution possible later: without knowing who said
        what and how heavily it counted, the self-improvement loop cannot tell
        which persona earned or lost the money.
        """
        return {
            "votes": [
                {
                    "persona": v.persona_key,
                    "action": v.action.value,
                    "confidence": str(v.confidence),
                    "weight": str(v.weight),
                    "effective": str(v.effective),
                    "lenses": [l.value for l in v.lenses],
                    "rationale": v.rationale[:500],
                    "changed_by": v.changed_by[:300],
                }
                for v in self.votes
            ],
            "abstentions": self.abstentions,
            "dispersion": str(self.dispersion),
            "lens_diversity": self.lens_diversity,
            "participating": self.participating,
            "escalated": self.escalate,
        }

    def summary(self) -> str:
        return (
            f"{self.action.value.upper()} conf={self.confidence:.2f} "
            f"(net {self.net_direction:+.2f}, {self.participating} voted, "
            f"{len(self.abstentions)} abstained, dispersion {self.dispersion:.2f}, "
            f"{self.lens_diversity} lenses"
            + (", ESCALATE" if self.escalate else "") + ")"
        )


class PersonaPanel:
    """Aggregates persona votes into a single decision."""

    def __init__(
        self,
        registry: PersonaRegistry,
        dispersion_escalation: Decimal = Decimal("0.4"),
        min_participants: int = 2,
        min_lens_diversity: int = 2,
    ) -> None:
        self.registry = registry
        self.dispersion_escalation = dispersion_escalation
        self.min_participants = min_participants
        self.min_lens_diversity = min_lens_diversity

    # -- weighting ---------------------------------------------------------

    def weight_for(self, spec: PersonaSpec, horizon: Horizon) -> Decimal:
        """Reliability times horizon fit.

        A persona operating far from its native timeframe counts for less even
        when technically applicable, and one the loop has found unreliable
        counts for less still.
        """
        return spec.reliability * spec.horizon_fit(horizon)

    # -- aggregation -------------------------------------------------------

    def aggregate(
        self,
        votes: Sequence[PersonaVote],
        asset_class: AssetClass,
        horizon: Horizon,
        extra_abstentions: Mapping[str, str] | None = None,
    ) -> PanelVerdict:
        """Combine votes into a verdict."""
        # Apply weights from the registry, so a caller cannot inflate a
        # persona's influence by setting the weight itself.
        weighted: list[PersonaVote] = []
        for vote in votes:
            try:
                spec = self.registry.get(vote.persona_key)
            except KeyError:
                log.warning("discarding vote from unknown persona %r", vote.persona_key)
                continue
            vote.weight = self.weight_for(spec, horizon)
            vote.lenses = vote.lenses or spec.lenses
            weighted.append(vote)

        abstentions = dict(
            self.registry.abstentions(asset_class, horizon)
        )
        if extra_abstentions:
            abstentions.update(extra_abstentions)
        for vote in weighted:
            if vote.abstained:
                abstentions.setdefault(
                    vote.persona_key,
                    vote.rationale[:200] or "abstained without a stated reason",
                )

        participating = [v for v in weighted if not v.abstained and v.weight > ZERO]
        verdict = PanelVerdict(
            action=Action.HOLD, confidence=ZERO, net_direction=ZERO,
            votes=weighted, abstentions=abstentions,
            participating=len(participating),
        )

        if not participating:
            return verdict

        # Abstentions are excluded from the denominator on purpose: counting
        # them as neutral would let inapplicable personas dilute informed ones
        # toward inaction.
        total_weight = sum((v.weight for v in participating), ZERO)
        if total_weight <= ZERO:
            return verdict

        net = sum((v.effective for v in participating), ZERO) / total_weight
        verdict.net_direction = net

        # Dispersion: weighted mean absolute deviation of direction, normalised.
        mean_direction = sum(
            (v.direction * v.weight for v in participating), ZERO
        ) / total_weight
        deviation = sum(
            (abs(v.direction - mean_direction) * v.weight for v in participating), ZERO
        ) / total_weight
        # Max possible deviation is 2 (one vote at +1, another at -1).
        verdict.dispersion = min(ONE, deviation / Decimal(2))

        # Lens diversity among those agreeing with the net direction.
        agreeing = [
            v for v in participating
            if (v.direction > ZERO) == (net > ZERO) and v.direction != ZERO
        ]
        lenses: set[Lens] = set()
        for vote in agreeing:
            lenses.update(vote.lenses)
        verdict.lens_diversity = len(lenses)

        if net > ZERO:
            verdict.action = Action.BUY
        elif net < ZERO:
            verdict.action = Action.SELL
        else:
            verdict.action = Action.HOLD

        confidence = abs(net)

        # Penalise agreement that comes from a single lens: five personas all
        # reading momentum is one opinion, not five.
        if verdict.lens_diversity < self.min_lens_diversity and len(agreeing) > 1:
            confidence *= Decimal("0.7")

        # Penalise a thin panel.
        if verdict.participating < self.min_participants:
            confidence *= Decimal("0.6")

        # Penalise genuine disagreement.
        confidence *= (ONE - verdict.dispersion * Decimal("0.5"))

        verdict.confidence = max(ZERO, min(ONE, confidence))

        verdict.escalate = (
            verdict.dispersion >= self.dispersion_escalation
            or verdict.lens_diversity < self.min_lens_diversity
        )
        return verdict

    # -- decision ----------------------------------------------------------

    def to_decision(
        self,
        verdict: PanelVerdict,
        instrument_key: str,
        horizon: Horizon,
        regime: str | None = None,
    ) -> Decision:
        """Turn a verdict into a persisted Decision with full attribution."""
        rationales = [
            f"{v.persona_key}: {v.rationale[:200]}"
            for v in verdict.voted
        ]
        return Decision(
            instrument_key=instrument_key,
            action=verdict.action,
            confidence=verdict.confidence,
            horizon=horizon.value,
            regime=regime,
            rationale=" | ".join(rationales) if rationales else "no persona voted",
            contributions=verdict.contributions(),
            metadata={"escalated": verdict.escalate},
        )


def abstain(persona_key: str, reason: str) -> PersonaVote:
    """Construct an explicit abstention.

    Preferred over omitting a vote: an abstention with a stated reason is
    information, whereas a missing vote is indistinguishable from a crashed
    agent.
    """
    return PersonaVote(
        persona_key=persona_key, action=Action.HOLD, confidence=ZERO,
        rationale=reason,
    )


__all__ = ["PanelVerdict", "PersonaPanel", "PersonaVote", "abstain"]
