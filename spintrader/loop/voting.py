"""Turning personas into votes -- the layer the panel needs but did not have.

:mod:`spintrader.agents.panel` aggregates :class:`PersonaVote`s into a verdict,
but nothing produced those votes. This module does: it asks each applicable
persona what it thinks of one instrument, right now, and returns a vote.

Two voters, chosen at runtime rather than at build time:

* :class:`LLMVoter` queries the persona's own methodology prompt against the
  GB10 model tiers, with a JSON grammar so the reply is a vote and not an essay.
  All calls in a cycle are one batch, and -- per the router's hard constraint --
  the batch is grouped by tier and run quick-then-deep, never interleaved, so
  the 99 GB deep model pages in at most once. Batching is across assets too, not
  just within one, because a tier switch costs ~35 s of reload.

* :class:`BootstrapVoter` needs no LLM at all. It derives a single quant opinion
  (trend sign, damped by regime risk) and returns it for every persona under one
  shared :class:`~spintrader.agents.personas.spec.Lens`. That single lens is the
  honest part: the panel then sees ``lens_diversity == 1``, discounts the
  confidence and raises ``escalate``, correctly reporting that this mandate is
  one signal wearing many hats rather than a genuine ensemble. It exists so the
  loop runs -- and is testable -- with the GB10 unreachable, and as the explicit
  fallback when an LLM call fails.

A failed or unparseable LLM reply becomes an *abstention with a reason*, never a
fabricated vote: a made-up opinion is worse than a missing one, and an
abstention with a cause is information the panel can use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Protocol, Sequence, runtime_checkable

from spintrader.agents.panel import PersonaVote, abstain
from spintrader.agents.personas.spec import Lens, PersonaSpec
from spintrader.core.types import Action, to_decimal
from spintrader.llm.router import LLMError, LLMRouter, Tier, extract_json
from spintrader.loop.context import MarketContext

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class VoteItem:
    """One (persona, instrument) question the panel needs answered."""
    spec: PersonaSpec
    context: MarketContext

    @property
    def instrument_key(self) -> str:
        return self.context.instrument_key


class Voter(Protocol):
    """Produces votes for a batch of (persona, instrument) questions.

    Batch-shaped on purpose: the LLM voter must group its calls by model tier,
    which it cannot do one persona at a time.
    """

    def vote_all(self, items: Sequence[VoteItem]) -> dict[str, list[PersonaVote]]:
        """Return votes grouped by instrument key."""
        ...


@runtime_checkable
class Adjudicator(Protocol):
    """A voter that can re-adjudicate contested instruments on the deep tier.

    Adaptive escalation (task 18) only fires for voters that implement this.
    The bootstrap voter has no deep model, so it is deliberately *not* an
    ``Adjudicator``; :class:`~spintrader.loop.mandate.MandateService` checks for
    this capability and degrades escalation to a graceful no-op when it is
    absent. Being a runtime-checkable Protocol, ``isinstance(voter, Adjudicator)``
    is a structural check for the ``adjudicate_all`` method, so a test fake needs
    only to define it.
    """

    def adjudicate_all(
        self, items: Sequence[VoteItem]
    ) -> dict[str, list[PersonaVote]]:
        """Re-vote the given (persona, instrument) questions on the deep tier."""
        ...


# --------------------------------------------------------------------------
# Bootstrap voter (no LLM)
# --------------------------------------------------------------------------

class BootstrapVoter:
    """A deterministic, single-signal voter for offline runs and tests.

    Every persona receives the same trend-following opinion, tagged with a
    single statistical lens so the panel's diversity machinery correctly marks
    the resulting mandate as low-conviction and escalatable. This is not a
    substitute for the LLM panel; it is a floor that lets the loop function
    without one.
    """

    #: A trend weaker than this (in absolute terms) is treated as no signal.
    def __init__(
        self,
        trend_threshold: float = 0.0005,
        max_confidence: Decimal = Decimal("0.75"),
    ) -> None:
        self.trend_threshold = trend_threshold
        self.max_confidence = to_decimal(max_confidence)

    def vote_all(self, items: Sequence[VoteItem]) -> dict[str, list[PersonaVote]]:
        out: dict[str, list[PersonaVote]] = {}
        for item in items:
            out.setdefault(item.instrument_key, []).append(self._vote(item))
        return out

    def _vote(self, item: VoteItem) -> PersonaVote:
        ctx = item.context
        trend = ctx.trend_strength
        if abs(trend) < self.trend_threshold:
            return abstain(item.spec.key, "no discernible trend (bootstrap)")

        action = Action.BUY if trend > 0 else Action.SELL
        # Confidence rises with trend clarity and falls with regime risk. Kept
        # well below 1 -- a mechanical trend sign is a weak opinion and should
        # not masquerade as a strong one.
        clarity = min(1.0, abs(trend) / 0.01)          # 1% trend -> full clarity
        damp = 1.0 - float(ctx.regime_risk) * 0.5
        confidence = self.max_confidence * to_decimal(max(0.0, clarity * damp))
        return PersonaVote(
            persona_key=item.spec.key,
            action=action,
            confidence=max(ZERO, min(ONE, confidence)),
            rationale=f"bootstrap: trend {trend:+.3%}, regime_risk {ctx.regime_risk}",
            changed_by="a trend reversal below the slow moving average",
            # One shared lens on purpose: the panel then penalises the lack of
            # diversity and flags escalation, which is the honest signal.
            lenses=(Lens.STATISTICAL,),
        )


# --------------------------------------------------------------------------
# LLM voter
# --------------------------------------------------------------------------

# The grammar Ollama constrains the reply to. Keeping it minimal makes the vote
# reliable; the panel does the reasoning-about-reasoning, not the model.
VOTE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "changed_by": {"type": "string"},
    },
    "required": ["action", "confidence", "rationale"],
}


# Appended to a persona's own system prompt on the deep re-adjudication pass.
# The methodology is unchanged -- the persona still reasons as itself -- but the
# deep model is told the decision was contested on the first pass and asked to
# weigh the strongest opposing evidence before committing. This is the "sharper
# adjudication instruction" of task 18.
DEEP_ADJUDICATION_DIRECTIVE = (
    "\n\n--- Deep re-adjudication ---\n"
    "This decision was contested on the first pass: the panel was split, or too "
    "few independent methods spoke to it. Re-examine it with more care than a "
    "routine call warrants. Identify the single strongest piece of evidence "
    "against your view and weigh it honestly before you commit; state it as "
    "changed_by. Commit to the direction your own methodology genuinely "
    "supports -- do not split the difference to appear balanced -- and abstain "
    "only if the evidence you require is truly absent, not merely because the "
    "decision is hard."
)


def _vote_prompt(item: VoteItem) -> str:
    ctx = item.context
    return (
        f"Decision: should the book be long, short, or flat in "
        f"{ctx.instrument.symbol} over a {ctx.horizon.value} horizon?\n\n"
        f"Market snapshot:\n  {ctx.summary()}\n\n"
        f"Respond with a JSON object: action (buy|sell|hold), confidence (0..1, "
        f"0 means abstain), rationale (one or two sentences), and changed_by "
        f"(the single piece of evidence that would most change your mind). "
        f"If the evidence your methodology needs is not present here, set "
        f"action=hold and confidence=0."
    )


class LLMVoter:
    """Queries persona methodologies against the GB10 model tiers.

    One :meth:`vote_all` call issues one batched request set on the quick tier.
    The panel decides afterwards whether a verdict is contentious enough to
    escalate; when it is, :class:`~spintrader.loop.mandate.MandateService` calls
    :meth:`adjudicate_all` to re-run just those instruments on the deep model.
    That is the adaptive escalation of task 18: this voter is the
    :class:`Adjudicator`, and the two passes are always separate batches, so the
    99 GB deep model pages in at most once per cycle and is never interleaved
    with quick calls.
    """

    def __init__(
        self,
        router: LLMRouter,
        tier: Tier = Tier.QUICK,
        deep_tier: Tier = Tier.DEEP,
        temperature: float | None = None,
        max_tokens: int = 400,
        adjudication_directive: str | None = None,
    ) -> None:
        self.router = router
        self.tier = tier
        self.deep_tier = deep_tier
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.adjudication_directive = (
            DEEP_ADJUDICATION_DIRECTIVE if adjudication_directive is None
            else adjudication_directive
        )

    def vote_all(self, items: Sequence[VoteItem]) -> dict[str, list[PersonaVote]]:
        """First-pass votes on the quick tier."""
        return self._vote_batch(items, self.tier)

    def adjudicate_all(
        self, items: Sequence[VoteItem]
    ) -> dict[str, list[PersonaVote]]:
        """Deep-tier re-adjudication of contested instruments (task 18).

        Reuses each persona's own methodology prompt, but routed to the deep
        model and extended with :data:`DEEP_ADJUDICATION_DIRECTIVE`. Runs as a
        single batched deep pass across every persona of every escalated
        instrument, so the deep model loads once for the whole escalation set.
        """
        return self._vote_batch(
            items, self.deep_tier, system_suffix=self.adjudication_directive,
        )

    def _vote_batch(
        self,
        items: Sequence[VoteItem],
        tier: Tier,
        system_suffix: str | None = None,
    ) -> dict[str, list[PersonaVote]]:
        out: dict[str, list[PersonaVote]] = {item.instrument_key: [] for item in items}
        if not items:
            return out

        requests = []
        for item in items:
            system = item.spec.system_prompt()
            if system_suffix:
                system += system_suffix
            requests.append({
                "prompt": _vote_prompt(item),
                "system": system,
                "json_schema": VOTE_SCHEMA,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "think": False,    # schema + reasoning interact badly; see router
            })

        results = self.router.run_batched({tier: requests})
        responses = results.get(tier, [])

        for item, response in zip(items, responses):
            out[item.instrument_key].append(self._to_vote(item, response))
        # Any item beyond the responses we got back (should not happen) abstains.
        for item in items[len(responses):]:
            out[item.instrument_key].append(
                abstain(item.spec.key, "no response returned for this persona")
            )
        return out

    def _to_vote(self, item: VoteItem, response) -> PersonaVote:
        if isinstance(response, Exception):
            return abstain(item.spec.key, f"llm error: {str(response)[:120]}")
        try:
            data = extract_json(response.text)
        except LLMError as exc:
            return abstain(item.spec.key, f"unparseable reply: {str(exc)[:120]}")

        try:
            action = Action(str(data.get("action", "hold")).lower())
        except ValueError:
            return abstain(item.spec.key, f"unknown action {data.get('action')!r}")

        confidence = to_decimal(data.get("confidence", 0))
        confidence = max(ZERO, min(ONE, confidence))
        return PersonaVote(
            persona_key=item.spec.key,
            action=action,
            confidence=confidence,
            rationale=str(data.get("rationale", ""))[:1000],
            changed_by=str(data.get("changed_by", ""))[:500],
            lenses=item.spec.lenses,       # a real vote keeps its own lenses
        )


__all__ = [
    "Adjudicator", "BootstrapVoter", "DEEP_ADJUDICATION_DIRECTIVE", "LLMVoter",
    "VOTE_SCHEMA", "VoteItem", "Voter",
]
