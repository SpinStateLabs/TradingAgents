"""Daily recalibration: moving persona reliability toward realised skill.

The attribution half of the loop (:mod:`spintrader.loop.attribution`) scores
each persona's forecasts with a proper scoring rule and accumulates a shrunk
:class:`~spintrader.loop.attribution.ReliabilityEstimate` per persona. This
module is the *governance* half that sits on top of it: once a day it reads
those estimates and decides, persona by persona, whether the reliability weight
the panel uses (:meth:`PersonaPanel.weight_for` reads ``spec.reliability``)
should actually move -- and if so, by how much.

Despite the name, there is no LLM and no network here. "Recalibration" means
re-deriving each persona's earned influence from realised attribution, and doing
it *slowly and defensibly*. The division of responsibility is deliberate:

* the tracker owns the **statistics** -- Bayesian shrinkage of raw Brier skill
  toward a neutral prior, which is what stops ten lucky calls from moving a
  weight;
* this agent owns the **policy** -- an evidence gate, a per-day step bound, and
  a floor, which is what stops a single bad week from silencing a persona and
  what keeps a human, not the loop, in charge of a permanent zero.

Why every guardrail exists
--------------------------
On this book a persona accumulates a handful of scored predictions a week, so a
weight that reacts quickly is a weight that fits noise. The three guardrails
each close one failure mode:

1. **Evidence gate.** Below :attr:`RecalibrationPolicy.min_observations` the
   persona is left completely untouched. Shrinkage already pulls a thin sample
   toward neutral, but the gate is a harder rule: no evidence, no change, not
   even a small one.
2. **Step bound.** A weight moves at most :attr:`RecalibrationPolicy.max_step`
   toward the skill-implied target in one day. Sustained skill still arrives at
   the target; it just takes several days, which is the point.
3. **Floor.** Reliability is clamped to ``[floor, 1]``. A persona is never
   automatically driven to zero: it may be the only one applicable to a
   decision, and a permanent silencing is a decision for a human to make.

Every ruling -- old weight, new weight, the evidence and skill behind it, and
*why* it did or did not move -- is recorded in a :class:`RecalibrationReport`
for the audit trail. If a weight moved, the report says by how much and which
guardrail bound it; if it did not, the report says why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Callable, Iterable

from spintrader.agents.personas.spec import PersonaRegistry
from spintrader.core.types import utcnow
from spintrader.loop.attribution import (
    MAX_STEP,
    MIN_RELIABILITY,
    PRIOR_SKILL,
    PRIOR_STRENGTH,
    ForecastOutcome,
    ReliabilityEstimate,
    ReliabilityTracker,
    skill_to_reliability,
)

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RecalibrationPolicy:
    """The guardrails the recalibration applies. Deliberately conservative.

    Defaults are inherited from the attribution module so that the governance
    layer and the statistical layer agree unless a caller deliberately diverges
    them.
    """

    # A persona is left untouched until it has at least this many scored
    # outcomes. Matches PRIOR_STRENGTH: a persona must be "established" before
    # its weight is allowed to move at all.
    min_observations: int = PRIOR_STRENGTH

    # Optional secondary gate on the fraction of the estimate driven by data
    # rather than the prior. 0 disables it; min_observations is the primary gate.
    min_evidence_weight: float = 0.0

    # Largest reliability change permitted in one recalibration.
    max_step: Decimal = MAX_STEP

    # Reliability is clamped to [floor, ceiling]. The floor is why a bad run
    # cannot zero a persona out; the ceiling is a plain sanity bound.
    floor: Decimal = MIN_RELIABILITY
    ceiling: Decimal = ONE

    # Estimate from the most recent N outcomes only (None == all history). A
    # persona's edge can decay, and an all-history mean hides that.
    window: int | None = None

    def __post_init__(self) -> None:
        if self.min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        if self.max_step <= ZERO:
            raise ValueError("max_step must be positive")
        if not (ZERO <= self.floor <= self.ceiling <= ONE):
            raise ValueError("require 0 <= floor <= ceiling <= 1")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

@dataclass(slots=True)
class PersonaRecalibration:
    """One persona's recalibration ruling, for the audit trail."""

    persona_key: str
    old_reliability: Decimal
    new_reliability: Decimal
    n_observations: int
    evidence_weight: float
    raw_skill: float                 # unshrunk mean Brier skill
    shrunk_skill: float              # after shrinkage toward the prior
    skill_target: Decimal            # reliability the skill implies, pre-guardrails
    hit_rate: float
    pnl_attributed: Decimal
    changed: bool
    step_capped: bool = False        # the per-day step bound bound the move
    floored: bool = False            # the floor lifted the target
    ceiled: bool = False             # the ceiling capped the target
    reason: str = ""

    @property
    def delta(self) -> Decimal:
        return self.new_reliability - self.old_reliability

    def summary(self) -> str:
        if not self.changed:
            return (
                f"{self.persona_key}: {self.old_reliability:.3f} unchanged "
                f"[n={self.n_observations}, skill {self.shrunk_skill:+.3f}] "
                f"-- {self.reason}"
            )
        flags = "".join(
            f for f, on in (
                (" [capped]", self.step_capped),
                (" [floor]", self.floored),
                (" [ceil]", self.ceiled),
            ) if on
        )
        return (
            f"{self.persona_key}: {self.old_reliability:.3f} -> "
            f"{self.new_reliability:.3f} ({self.delta:+.3f}){flags} "
            f"[n={self.n_observations}, skill {self.shrunk_skill:+.3f}, "
            f"target {self.skill_target:.3f}]"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "persona": self.persona_key,
            "old_reliability": str(self.old_reliability),
            "new_reliability": str(self.new_reliability),
            "delta": str(self.delta),
            "n_observations": self.n_observations,
            "evidence_weight": self.evidence_weight,
            "raw_skill": self.raw_skill,
            "shrunk_skill": self.shrunk_skill,
            "skill_target": str(self.skill_target),
            "hit_rate": self.hit_rate,
            "pnl_attributed": str(self.pnl_attributed),
            "changed": self.changed,
            "step_capped": self.step_capped,
            "floored": self.floored,
            "ceiled": self.ceiled,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RecalibrationReport:
    """The full outcome of one recalibration pass.

    ``entries`` covers every persona that had scored outcomes *and* is in the
    registry. Registry personas with no outcomes are trivially unchanged and are
    listed in ``skipped_no_data``; tracked personas the registry does not know
    are listed in ``skipped_unknown`` rather than silently dropped.
    """

    generated_at: datetime
    entries: list[PersonaRecalibration] = field(default_factory=list)
    skipped_no_data: list[str] = field(default_factory=list)
    skipped_unknown: list[str] = field(default_factory=list)
    policy: RecalibrationPolicy = field(default_factory=RecalibrationPolicy)
    narrative: str | None = None

    @property
    def changed(self) -> list[PersonaRecalibration]:
        return [e for e in self.entries if e.changed]

    @property
    def unchanged(self) -> list[PersonaRecalibration]:
        return [e for e in self.entries if not e.changed]

    def get(self, persona_key: str) -> PersonaRecalibration | None:
        for entry in self.entries:
            if entry.persona_key == persona_key:
                return entry
        return None

    def summary(self) -> str:
        moved = self.changed
        head = (
            f"recalibration @ {self.generated_at.isoformat()}: "
            f"{len(moved)} of {len(self.entries)} personas moved"
        )
        if self.skipped_no_data:
            head += f", {len(self.skipped_no_data)} without evidence"
        if self.skipped_unknown:
            head += f", {len(self.skipped_unknown)} unknown"
        return head

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "entries": [e.as_dict() for e in self.entries],
            "skipped_no_data": list(self.skipped_no_data),
            "skipped_unknown": list(self.skipped_unknown),
            "policy": {
                "min_observations": self.policy.min_observations,
                "min_evidence_weight": self.policy.min_evidence_weight,
                "max_step": str(self.policy.max_step),
                "floor": str(self.policy.floor),
                "ceiling": str(self.policy.ceiling),
                "window": self.policy.window,
            },
            "narrative": self.narrative,
        }


# A narrator turns a finished report into human-readable prose. Fully optional
# and injected, so an LLM-backed one can be supplied without this module ever
# depending on a model or a network.
Narrator = Callable[["RecalibrationReport"], str]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class RecalibrationAgent:
    """Recalibrates a persona registry's reliabilities from realised attribution.

    Deterministic and offline. Construct once with a policy (and optionally a
    narrator), then call :meth:`run` each day with the live registry and the
    reliability tracker that has been accumulating the day's resolved outcomes.
    """

    def __init__(
        self,
        policy: RecalibrationPolicy | None = None,
        narrator: Narrator | None = None,
    ) -> None:
        self.policy = policy or RecalibrationPolicy()
        self.narrator = narrator

    def run(
        self,
        registry: PersonaRegistry,
        tracker: ReliabilityTracker,
        *,
        window: int | None = None,
        now: datetime | None = None,
    ) -> RecalibrationReport:
        """Recalibrate ``registry`` in place from ``tracker`` and report on it.

        Only personas that clear the evidence gate are written back to the
        registry; everything else is left exactly as it was. The returned report
        records every ruling, including the ones that changed nothing.
        """
        now = now or utcnow()
        window = window if window is not None else self.policy.window

        reg_keys = set(registry.keys())
        current = {key: registry.get(key).reliability for key in reg_keys}

        # estimate_all gives shrunk skill per tracked persona in one pass. We
        # pass the live weights as ``current`` so each estimate's ``previous``
        # is honest, but the applied value is (re)derived here under this
        # agent's policy, not the tracker's -- governance stays with the agent.
        estimates = tracker.estimate_all(current=current, window=window)
        tracked = set(estimates)

        entries: list[PersonaRecalibration] = []
        skipped_unknown: list[str] = []
        for key in sorted(tracked):
            if key not in reg_keys:
                skipped_unknown.append(key)
                continue
            entry = self._recalibrate_one(key, current[key], estimates[key])
            if entry.changed:
                registry.update_reliability(key, entry.new_reliability)
            entries.append(entry)

        report = RecalibrationReport(
            generated_at=now,
            entries=entries,
            skipped_no_data=sorted(reg_keys - tracked),
            skipped_unknown=skipped_unknown,
            policy=self.policy,
        )
        if self.narrator is not None:
            report.narrative = self.narrator(report)

        log.info("%s", report.summary())
        return report

    def run_from_outcomes(
        self,
        registry: PersonaRegistry,
        outcomes: Iterable[ForecastOutcome],
        *,
        window: int | None = None,
        now: datetime | None = None,
        prior_strength: int = PRIOR_STRENGTH,
        prior_skill: float = PRIOR_SKILL,
    ) -> RecalibrationReport:
        """Convenience path: recalibrate from a raw batch of outcomes.

        Builds a fresh tracker from ``outcomes`` and recalibrates against it.
        Note this discards any history the batch does not contain, so the
        accumulating-tracker path (:meth:`run`) is preferred in production; this
        exists for one-shot use and for testing.
        """
        tracker = ReliabilityTracker(
            prior_strength=prior_strength,
            prior_skill=prior_skill,
            max_step=self.policy.max_step,
            min_reliability=self.policy.floor,
        )
        tracker.record_many(outcomes)
        return self.run(registry, tracker, window=window, now=now)

    # -- per-persona ruling -----------------------------------------------

    def _recalibrate_one(
        self,
        key: str,
        current: Decimal,
        est: ReliabilityEstimate,
    ) -> PersonaRecalibration:
        p = self.policy
        skill_target = skill_to_reliability(est.shrunk_skill)

        entry = PersonaRecalibration(
            persona_key=key,
            old_reliability=current,
            new_reliability=current,
            n_observations=est.n_observations,
            evidence_weight=est.evidence_weight,
            raw_skill=est.raw_skill,
            shrunk_skill=est.shrunk_skill,
            skill_target=skill_target,
            hit_rate=est.hit_rate,
            pnl_attributed=est.pnl_attributed,
            changed=False,
        )

        # --- evidence gate: no evidence, no change ------------------------
        if est.n_observations < p.min_observations:
            entry.reason = (
                f"unchanged: {est.n_observations} outcomes below the "
                f"{p.min_observations} evidence threshold"
            )
            return entry
        if est.evidence_weight < p.min_evidence_weight:
            entry.reason = (
                f"unchanged: evidence weight {est.evidence_weight:.2f} below "
                f"the {p.min_evidence_weight:.2f} threshold"
            )
            return entry

        # --- step bound: move at most max_step toward the target ----------
        delta = skill_target - current
        step_capped = abs(delta) > p.max_step
        if step_capped:
            stepped = current + (p.max_step if delta > ZERO else -p.max_step)
        else:
            stepped = skill_target

        # --- floor / ceiling clamp ---------------------------------------
        floored = stepped < p.floor
        ceiled = stepped > p.ceiling
        new = max(p.floor, min(p.ceiling, stepped))

        entry.new_reliability = new
        entry.step_capped = step_capped
        entry.floored = floored
        entry.ceiled = ceiled
        entry.changed = new != current
        entry.reason = self._describe(entry, skill_target)
        return entry

    @staticmethod
    def _describe(entry: PersonaRecalibration, skill_target: Decimal) -> str:
        if not entry.changed:
            return "unchanged: already at the skill-implied target"
        direction = "gained" if entry.delta > ZERO else "lost"
        bits = [
            f"{direction} weight {entry.old_reliability:.3f}->"
            f"{entry.new_reliability:.3f} toward skill target {skill_target:.3f}"
        ]
        if entry.step_capped:
            bits.append("bounded by the daily step")
        if entry.floored:
            bits.append("held at the floor")
        if entry.ceiled:
            bits.append("held at the ceiling")
        return "; ".join(bits)


# --------------------------------------------------------------------------
# Functional entry point
# --------------------------------------------------------------------------

def recalibrate(
    registry: PersonaRegistry,
    tracker: ReliabilityTracker | None = None,
    *,
    outcomes: Iterable[ForecastOutcome] | None = None,
    policy: RecalibrationPolicy | None = None,
    window: int | None = None,
    narrator: Narrator | None = None,
    now: datetime | None = None,
) -> RecalibrationReport:
    """Recalibrate ``registry`` from realised attribution and return the report.

    Provide either a ``tracker`` (the production path -- it accumulates outcomes
    across days and shrinks them) or a raw batch of ``outcomes``. The registry's
    reliabilities are updated in place for personas that clear the guardrails,
    which is what next-cycle panel weighting will read.
    """
    agent = RecalibrationAgent(policy=policy, narrator=narrator)
    if tracker is not None:
        if outcomes is not None:
            raise ValueError("pass either a tracker or outcomes, not both")
        return agent.run(registry, tracker, window=window, now=now)
    if outcomes is not None:
        return agent.run_from_outcomes(registry, outcomes, window=window, now=now)
    raise ValueError("recalibrate requires either a tracker or outcomes")


def default_narrative(report: RecalibrationReport) -> str:
    """A deterministic, offline narrator, usable as the injected ``narrator``.

    Demonstrates the narrative slot without any LLM: an LLM-backed narrator with
    the same ``(report) -> str`` shape can be dropped in unchanged.
    """
    moved = report.changed
    if not moved:
        return (
            "No persona weights moved this cycle: "
            f"{len(report.unchanged)} evaluated, none cleared the guardrails "
            f"with enough of a skill gap to act on."
        )
    lines = [f"{len(moved)} persona weight(s) recalibrated from realised skill:"]
    for entry in sorted(moved, key=lambda e: e.delta):
        lines.append(f"  - {entry.summary()}")
    return "\n".join(lines)


__all__ = [
    "Narrator",
    "PersonaRecalibration",
    "RecalibrationAgent",
    "RecalibrationPolicy",
    "RecalibrationReport",
    "default_narrative",
    "recalibrate",
]
