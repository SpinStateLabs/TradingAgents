"""Persona specifications.

A persona here is a **methodology archetype**, named for its best-known
exponent. It is not a simulation of a person, does not claim to represent
anyone's current views, and should not be read as an endorsement by them. What
is modelled is a documented, publicly-described approach to deciding what to
own and when to stop owning it.

Why specs and not prompts
-------------------------
Each persona is structured data, not a blob of prompt text. That buys three
things a prompt cannot:

* **Attribution.** The self-improvement loop scores each persona's
  contribution separately, so one that reasons beautifully and loses money
  loses influence. A prompt blob cannot be weighted.
* **Abstention.** Personas declare which asset classes and horizons they can
  actually speak to. A quality-compounder methodology has nothing to say about
  an hourly Bitcoin decision, and forcing it to produce a number anyway
  manufactures a vote out of nothing. Abstention is a first-class output.
* **Versioning.** A spec can be diffed, so a change in behaviour is traceable
  to a change in the spec rather than to prompt drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from typing import Mapping, Sequence

from spintrader.core.types import AssetClass

ZERO = Decimal("0")
ONE = Decimal("1")


class Horizon(str, Enum):
    """Holding period a methodology is designed for.

    Ordered, so applicability can be checked as a range. The measured cost
    arithmetic on this book makes anything below INTRADAY untradeable at 26bps
    per side, and personas whose native horizon is YEARS cannot express
    themselves in an hourly decision.
    """
    MINUTES = "minutes"
    INTRADAY = "intraday"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"

    @property
    def rank(self) -> int:
        return _HORIZON_ORDER.index(self)


_HORIZON_ORDER = [
    Horizon.MINUTES, Horizon.INTRADAY, Horizon.DAYS,
    Horizon.WEEKS, Horizon.MONTHS, Horizon.YEARS,
]


class Lens(str, Enum):
    """The primary evidence a methodology reasons from."""
    FUNDAMENTAL = "fundamental"        # cash flows, balance sheets, valuation
    FORENSIC = "forensic"              # accounting quality, hidden liabilities
    MACRO = "macro"                    # rates, credit, policy, growth
    STATISTICAL = "statistical"        # price/volume patterns, weak signals
    TAIL = "tail"                      # distribution shape, convexity
    FLOW = "flow"                      # positioning, liquidity, leverage
    NARRATIVE = "narrative"            # sentiment, reflexivity, headlines
    STRUCTURAL = "structural"          # moats, monopoly, technology shifts


@dataclass(frozen=True, slots=True)
class PersonaSpec:
    """A methodology archetype.

    ``system_prompt`` is generated from these fields rather than stored, so the
    behaviour follows the spec and cannot silently diverge from it.
    """
    key: str
    display_name: str
    attribution: str                 # whose documented approach this models
    thesis: str                      # the core claim in one sentence
    method: tuple[str, ...]           # concrete steps it actually performs
    lenses: tuple[Lens, ...]
    native_horizon: Horizon
    min_horizon: Horizon
    max_horizon: Horizon
    asset_classes: frozenset[AssetClass]

    # Behavioural parameters
    contrarian: Decimal = Decimal("0.5")      # 0 trend-following .. 1 contrarian
    concentration: Decimal = Decimal("0.5")   # 0 diversified .. 1 concentrated
    patience: Decimal = Decimal("0.5")        # willingness to wait / hold through pain
    conviction_threshold: Decimal = Decimal("0.6")  # abstains below this

    # What would make this persona wrong. Stated explicitly, because a
    # methodology with no falsification condition cannot be evaluated and will
    # rationalise any outcome.
    invalidation: tuple[str, ...] = ()
    known_failure_modes: tuple[str, ...] = ()

    # Set by the self-improvement loop from realised attribution.
    reliability: Decimal = ONE

    def applies_to(
        self, asset_class: AssetClass, horizon: Horizon,
    ) -> tuple[bool, str]:
        """Whether this persona can speak to a decision, and why not if it cannot.

        Returning the reason matters: an abstention with a stated cause is
        information the panel can use, while a silent zero looks like a neutral
        opinion.
        """
        if asset_class not in self.asset_classes:
            return False, (
                f"{self.display_name} reasons about "
                f"{sorted(c.value for c in self.asset_classes)}, not "
                f"{asset_class.value}"
            )
        if horizon.rank < self.min_horizon.rank:
            return False, (
                f"{self.display_name} needs at least a {self.min_horizon.value} "
                f"horizon; this decision is {horizon.value}"
            )
        if horizon.rank > self.max_horizon.rank:
            return False, (
                f"{self.display_name} does not extend beyond "
                f"{self.max_horizon.value}; this decision is {horizon.value}"
            )
        return True, ""

    def horizon_fit(self, horizon: Horizon) -> Decimal:
        """How well a horizon suits this methodology, 0..1.

        Full weight at the native horizon, decaying with distance. A persona
        operating far from its natural timeframe should carry less weight even
        when technically applicable -- a months-horizon value process asked
        about a days-horizon decision is being stretched.
        """
        distance = abs(horizon.rank - self.native_horizon.rank)
        return max(ZERO, ONE - Decimal(distance) * Decimal("0.25"))

    def system_prompt(self) -> str:
        """Build the persona's instruction text from its spec."""
        lines = [
            f"You are a trading analyst applying the documented methodology of "
            f"{self.attribution}.",
            "",
            f"Core thesis: {self.thesis}",
            "",
            "Your method, in order:",
        ]
        lines += [f"  {i}. {step}" for i, step in enumerate(self.method, 1)]
        lines += [
            "",
            f"Primary evidence: {', '.join(l.value for l in self.lenses)}.",
            f"Natural holding period: {self.native_horizon.value}.",
        ]
        if self.invalidation:
            lines += ["", "You are WRONG if any of these hold:"]
            lines += [f"  - {item}" for item in self.invalidation]
        if self.known_failure_modes:
            lines += ["", "Known weaknesses of this methodology, which you must "
                          "weigh honestly rather than dismiss:"]
            lines += [f"  - {item}" for item in self.known_failure_modes]
        lines += [
            "",
            "Rules:",
            "  - If the evidence you need is unavailable, ABSTAIN. Say so "
            "plainly and set confidence to 0. A fabricated opinion is worse "
            "than no opinion.",
            "  - Do not adopt another methodology's reasoning to reach a view. "
            "Your value to the panel is that you are different from it.",
            "  - State the single piece of evidence that would most change "
            "your mind.",
        ]
        return "\n".join(lines)

    def with_reliability(self, reliability: Decimal) -> "PersonaSpec":
        """Return a copy with an updated reliability weight."""
        return replace(self, reliability=max(ZERO, min(ONE, reliability)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

class PersonaRegistry:
    """Named collection of personas, with applicability filtering."""

    def __init__(self, specs: Sequence[PersonaSpec] = ()) -> None:
        self._specs: dict[str, PersonaSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: PersonaSpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"persona {spec.key!r} is already registered")
        self._specs[spec.key] = spec

    def get(self, key: str) -> PersonaSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(
                f"unknown persona {key!r}; registered: {sorted(self._specs)}"
            ) from None

    def all(self) -> list[PersonaSpec]:
        return list(self._specs.values())

    def keys(self) -> list[str]:
        return sorted(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def applicable(
        self, asset_class: AssetClass, horizon: Horizon,
        min_reliability: Decimal = ZERO,
    ) -> list[PersonaSpec]:
        """Personas that can speak to this decision, best-fitting first."""
        eligible = [
            spec for spec in self._specs.values()
            if spec.applies_to(asset_class, horizon)[0]
            and spec.reliability >= min_reliability
        ]
        eligible.sort(
            key=lambda s: (s.horizon_fit(horizon) * s.reliability),
            reverse=True,
        )
        return eligible

    def abstentions(
        self, asset_class: AssetClass, horizon: Horizon,
    ) -> dict[str, str]:
        """Personas that cannot speak here, with the reason for each."""
        out: dict[str, str] = {}
        for spec in self._specs.values():
            ok, reason = spec.applies_to(asset_class, horizon)
            if not ok:
                out[spec.key] = reason
        return out

    def update_reliability(self, key: str, reliability: Decimal) -> None:
        self._specs[key] = self.get(key).with_reliability(reliability)

    def lens_coverage(
        self, asset_class: AssetClass, horizon: Horizon,
    ) -> dict[Lens, int]:
        """How many applicable personas use each lens.

        An ensemble whose members all reason from the same evidence is one
        opinion wearing several hats. This is the diagnostic for that.
        """
        counts: dict[Lens, int] = {}
        for spec in self.applicable(asset_class, horizon):
            for lens in spec.lenses:
                counts[lens] = counts.get(lens, 0) + 1
        return counts


__all__ = ["Horizon", "Lens", "PersonaRegistry", "PersonaSpec"]
