"""Attribution and reliability: deciding which personas have earned influence.

Two separate questions, deliberately not conflated.

**Was the forecast good?** Scored with a proper scoring rule (Brier) against the
realised direction. This drives reliability, because a persona does not choose
its position size -- the risk engine does -- and penalising a correct call that
was sized small, or rewarding a wrong call that happened to be sized large,
teaches the loop the wrong lesson.

**How much money did it make?** Tracked separately, for accounting and for
reporting. Useful, but a noisy basis for weighting: one large winning trade can
dominate fifty correct small calls.

Why shrinkage is not optional
-----------------------------
On this book -- roughly four assets, hourly-to-daily decisions -- a persona
accumulates a handful of scored predictions per week. A raw hit rate over ten
observations is almost pure noise: five correct calls out of ten is consistent
with anything from a useless persona to a good one. Moving weights on that
basis produces oscillation that looks like adaptation.

Every reliability estimate is therefore shrunk toward a neutral prior with a
strength of :data:`PRIOR_STRENGTH` pseudo-observations, and per-update movement
is capped. A persona needs sustained evidence to gain or lose influence, and
:class:`ReliabilityTracker.stability` exposes whether weights are converging or
thrashing.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from spintrader.core.types import Action, to_decimal, utcnow

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")
HALF = Decimal("0.5")

# Pseudo-observations of the prior. At 30, a persona needs roughly 30 scored
# predictions before the evidence outweighs the prior -- about a month of daily
# decisions on four assets. Lower values make the loop twitchy; higher values
# make it unable to learn at all.
PRIOR_STRENGTH = 30

# Neutral prior skill. 0.5 means "no information either way" once mapped to
# reliability, so a new persona starts trusted-but-unproven rather than at zero.
PRIOR_SKILL = 0.5

# Largest reliability change permitted in one update. Prevents a single bad week
# from silencing a persona outright.
MAX_STEP = Decimal("0.10")

# Floor on reliability. A persona is never fully silenced automatically: it may
# be the only one applicable to a decision, and a permanent zero is a decision
# for a human to make.
MIN_RELIABILITY = Decimal("0.05")


# --------------------------------------------------------------------------
# Scoring a single forecast
# --------------------------------------------------------------------------

def direction_to_probability(direction: Decimal) -> float:
    """Map a signed conviction in [-1, 1] to P(price rises) in [0, 1]."""
    d = max(-1.0, min(1.0, float(direction)))
    return (d + 1.0) / 2.0


def brier_score(probability: float, rose: bool) -> float:
    """Squared error of a probabilistic forecast. Lower is better, 0 is perfect.

    Brier is a *proper* scoring rule: it is minimised by reporting one's true
    belief, so a persona cannot improve its score by exaggerating confidence.
    That property is why it is used here rather than a raw hit rate, which
    rewards confident guessing.
    """
    outcome = 1.0 if rose else 0.0
    return (probability - outcome) ** 2


def brier_skill(probability: float, rose: bool) -> float:
    """Brier score rescaled so that an uninformative forecast scores 0.

    A 0.5 forecast scores exactly 0. A perfect confident call scores 1. A
    confidently wrong call scores -3, which is intentionally asymmetric: being
    loudly wrong should cost more than being quietly right earns.
    """
    return 1.0 - brier_score(probability, rose) / 0.25


@dataclass(slots=True)
class ForecastOutcome:
    """One persona's scored prediction on one resolved decision."""
    decision_id: str
    persona_key: str
    instrument_key: str
    direction: Decimal            # signed conviction at decision time
    weight: Decimal               # how heavily it counted
    realised_return: Decimal      # over the decision's horizon
    resolved_at: datetime
    pnl_attributed: Decimal = ZERO

    @property
    def rose(self) -> bool:
        return self.realised_return > ZERO

    @property
    def probability(self) -> float:
        return direction_to_probability(self.direction)

    @property
    def skill(self) -> float:
        return brier_skill(self.probability, self.rose)

    @property
    def was_directionally_right(self) -> bool:
        if self.direction == ZERO:
            return False
        return (self.direction > ZERO) == self.rose


# --------------------------------------------------------------------------
# P&L attribution
# --------------------------------------------------------------------------

def attribute_pnl(
    realised_pnl: Decimal,
    votes: Sequence[tuple[str, Decimal, Decimal]],
) -> dict[str, Decimal]:
    """Split realised P&L across the personas that ADVOCATED the position taken.

    ``votes`` is a sequence of ``(persona_key, direction, weight)``.

    Only advocates own the outcome. A persona that argued against the position
    and was overruled receives zero, not a debit -- it did not cause the trade,
    and debiting it for a loss it warned about would be perverse. Its
    correctness is captured by the Brier skill score, which is what drives
    reliability; this function is pure accounting for who asked for the trade.

    The position's direction is inferred from the weighted net of the votes, not
    from the sign of the P&L. Those are different things: a losing long tells
    you the market fell, not that the long was unadvocated, and conflating them
    credits dissenters and advocates identically.

    Attributions sum to ``realised_pnl``, so the ledger reconciles.
    """
    contributions = [(key, d * w) for key, d, w in votes if d != ZERO and w > ZERO]
    if not contributions:
        return {}

    net = sum((c for _, c in contributions), ZERO)
    if net == ZERO:
        # No net position was advocated; nobody owns the outcome.
        return {}

    position_is_long = net > ZERO
    advocates = [
        (key, abs(c)) for key, c in contributions
        if (c > ZERO) == position_is_long
    ]
    total = sum((c for _, c in advocates), ZERO)
    if total <= ZERO:
        return {}

    return {key: realised_pnl * (conviction / total) for key, conviction in advocates}


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------

@dataclass(slots=True)
class ReliabilityEstimate:
    """A persona's reliability with the evidence behind it."""
    persona_key: str
    reliability: Decimal
    n_observations: int
    raw_skill: float                  # unshrunk mean Brier skill
    shrunk_skill: float
    hit_rate: float
    pnl_attributed: Decimal
    previous: Decimal | None = None
    clamped: bool = False             # whether MAX_STEP bound the change

    @property
    def evidence_weight(self) -> float:
        """Fraction of the estimate driven by data rather than the prior."""
        return self.n_observations / (self.n_observations + PRIOR_STRENGTH)

    @property
    def is_established(self) -> bool:
        """Whether there is enough data to take this seriously."""
        return self.n_observations >= PRIOR_STRENGTH

    def summary(self) -> str:
        arrow = ""
        if self.previous is not None:
            delta = self.reliability - self.previous
            arrow = f" ({delta:+.3f})" + (" [capped]" if self.clamped else "")
        status = "established" if self.is_established else "provisional"
        return (
            f"{self.persona_key}: {self.reliability:.3f}{arrow} "
            f"[{status}, n={self.n_observations}, skill {self.raw_skill:+.3f}, "
            f"hit {self.hit_rate:.0%}, pnl {self.pnl_attributed:+.2f}]"
        )


def skill_to_reliability(skill: float) -> Decimal:
    """Map a Brier skill score to a reliability multiplier in [0, 1].

    Skill runs from -3 (confidently wrong) through 0 (uninformative) to 1
    (perfect). Reliability maps 0 skill to 0.5 rather than to 0: an
    uninformative persona still deserves a voice in the ensemble, since its
    disagreement carries information even when its direction does not.
    """
    clamped = max(-1.0, min(1.0, skill))
    return to_decimal(round((clamped + 1.0) / 2.0, 6))


class ReliabilityTracker:
    """Accumulates outcomes and produces shrunk reliability estimates."""

    def __init__(
        self,
        prior_strength: int = PRIOR_STRENGTH,
        prior_skill: float = PRIOR_SKILL,
        max_step: Decimal = MAX_STEP,
        min_reliability: Decimal = MIN_RELIABILITY,
    ) -> None:
        self.prior_strength = prior_strength
        self.prior_skill = prior_skill
        self.max_step = max_step
        self.min_reliability = min_reliability
        self._outcomes: dict[str, list[ForecastOutcome]] = {}
        self._history: dict[str, list[Decimal]] = {}

    # -- ingestion ---------------------------------------------------------

    def record(self, outcome: ForecastOutcome) -> None:
        self._outcomes.setdefault(outcome.persona_key, []).append(outcome)

    def record_many(self, outcomes: Iterable[ForecastOutcome]) -> int:
        count = 0
        for outcome in outcomes:
            self.record(outcome)
            count += 1
        return count

    def outcomes_for(self, persona_key: str) -> list[ForecastOutcome]:
        return list(self._outcomes.get(persona_key, ()))

    # -- estimation --------------------------------------------------------

    def estimate(
        self,
        persona_key: str,
        current: Decimal | None = None,
        window: int | None = None,
    ) -> ReliabilityEstimate:
        """Shrunk reliability for one persona.

        ``window`` limits the estimate to the most recent N outcomes, which
        matters because a persona's edge can decay: a methodology that worked
        in one regime may not in the next, and an all-history mean hides that.
        """
        outcomes = self._outcomes.get(persona_key, [])
        if window is not None:
            outcomes = outcomes[-window:]

        n = len(outcomes)
        if n == 0:
            reliability = current if current is not None else ONE
            return ReliabilityEstimate(
                persona_key=persona_key, reliability=reliability,
                n_observations=0, raw_skill=self.prior_skill,
                shrunk_skill=self.prior_skill, hit_rate=0.0,
                pnl_attributed=ZERO, previous=current,
            )

        skills = [o.skill for o in outcomes]
        raw_skill = statistics.fmean(skills)

        # Bayesian shrinkage toward the prior. With n small, the prior
        # dominates; only sustained evidence moves the estimate.
        shrunk = (
            (n * raw_skill + self.prior_strength * self.prior_skill)
            / (n + self.prior_strength)
        )

        target = skill_to_reliability(shrunk)
        hit_rate = sum(1 for o in outcomes if o.was_directionally_right) / n
        pnl = sum((o.pnl_attributed for o in outcomes), ZERO)

        clamped = False
        if current is not None:
            delta = target - current
            if abs(delta) > self.max_step:
                target = current + (self.max_step if delta > ZERO else -self.max_step)
                clamped = True

        target = max(self.min_reliability, min(ONE, target))
        self._history.setdefault(persona_key, []).append(target)

        return ReliabilityEstimate(
            persona_key=persona_key, reliability=target, n_observations=n,
            raw_skill=raw_skill, shrunk_skill=shrunk, hit_rate=hit_rate,
            pnl_attributed=pnl, previous=current, clamped=clamped,
        )

    def estimate_all(
        self,
        current: Mapping[str, Decimal] | None = None,
        window: int | None = None,
    ) -> dict[str, ReliabilityEstimate]:
        current = current or {}
        return {
            key: self.estimate(key, current.get(key), window=window)
            for key in sorted(self._outcomes)
        }

    # -- health ------------------------------------------------------------

    def stability(self, persona_key: str, lookback: int = 10) -> float:
        """How steady a persona's reliability has been. 1 is perfectly stable.

        Oscillation is the signature of fitting noise. A tracker whose weights
        swing every update is not adapting, it is chasing, and this is the
        metric that makes that visible rather than letting it pass for
        responsiveness.
        """
        history = self._history.get(persona_key, [])[-lookback:]
        if len(history) < 3:
            return 1.0
        deltas = [abs(float(b - a)) for a, b in zip(history, history[1:])]
        mean_delta = statistics.fmean(deltas)
        # Scaled against max_step: moving the full cap every update is total
        # instability.
        return max(0.0, 1.0 - mean_delta / float(self.max_step))

    def health(self, lookback: int = 10) -> dict[str, object]:
        """Loop-level diagnostics."""
        keys = sorted(self._outcomes)
        stabilities = {k: self.stability(k, lookback) for k in keys}
        counts = {k: len(self._outcomes[k]) for k in keys}
        established = [k for k, n in counts.items() if n >= self.prior_strength]
        return {
            "personas_tracked": len(keys),
            "established": established,
            "provisional": [k for k in keys if k not in established],
            "observations": counts,
            "stability": stabilities,
            "least_stable": min(stabilities, key=stabilities.get) if stabilities else None,
            "mean_stability": statistics.fmean(stabilities.values()) if stabilities else 1.0,
        }


# --------------------------------------------------------------------------
# Resolving decisions into outcomes
# --------------------------------------------------------------------------

def resolve_decision(
    decision_id: str,
    instrument_key: str,
    contributions: Mapping[str, object],
    entry_price: Decimal,
    exit_price: Decimal,
    resolved_at: datetime,
    realised_pnl: Decimal = ZERO,
) -> list[ForecastOutcome]:
    """Turn a stored decision plus its outcome into per-persona outcomes.

    ``contributions`` is the dict written to ``decisions.contributions`` by
    :meth:`PanelVerdict.contributions`. Reading it back is what closes the loop:
    without the vote record there is no way to know who said what, and
    attribution becomes impossible after the fact.
    """
    if entry_price <= ZERO:
        raise ValueError(f"invalid entry price {entry_price} for {decision_id}")

    realised_return = (exit_price - entry_price) / entry_price

    votes = contributions.get("votes") or []
    triples: list[tuple[str, Decimal, Decimal]] = []
    for vote in votes:                                  # type: ignore[union-attr]
        try:
            key = vote["persona"]
            action = vote["action"]
            confidence = to_decimal(vote["confidence"])
            weight = to_decimal(vote["weight"])
        except (KeyError, TypeError, ValueError):
            log.debug("skipping malformed vote record in %s", decision_id)
            continue

        if action == Action.BUY.value:
            direction = confidence
        elif action in (Action.SELL.value, Action.CLOSE.value):
            direction = -confidence
        else:
            continue                                    # abstained
        triples.append((key, direction, weight))

    pnl_split = attribute_pnl(realised_pnl, triples) if realised_pnl != ZERO else {}

    return [
        ForecastOutcome(
            decision_id=decision_id,
            persona_key=key,
            instrument_key=instrument_key,
            direction=direction,
            weight=weight,
            realised_return=realised_return,
            resolved_at=resolved_at,
            pnl_attributed=pnl_split.get(key, ZERO),
        )
        for key, direction, weight in triples
    ]


__all__ = [
    "MAX_STEP", "MIN_RELIABILITY", "PRIOR_SKILL", "PRIOR_STRENGTH",
    "ForecastOutcome", "ReliabilityEstimate", "ReliabilityTracker",
    "attribute_pnl", "brier_score", "brier_skill", "direction_to_probability",
    "resolve_decision", "skill_to_reliability",
]
