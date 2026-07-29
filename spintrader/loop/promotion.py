"""The promotion gate: what a candidate must prove before it trades money.

This is the component that decides whether the self-improvement loop actually
improves anything or merely churns. Its whole job is to say no.

The loop is a search process. It will generate candidate strategies, parameter
sets and agents, and it will find some that look good on historical data --
because with enough candidates, something always does. Demonstrated on this
project's own data: forty coin-flip strategies on real BTC history produced
Sharpe ratios from -0.55 to +1.05, and the best of them reads as significant
unless the number of trials is accounted for.

Every gate below exists to close one route by which noise reaches live capital:

1. **Cumulative trial accounting.** The trial count is the loop's lifetime
   total, not the current round's. Resetting it per round would let a hundred
   rounds of ten candidates each pass as ten independent tests.
2. **Deflated Sharpe.** The candidate must beat what selection alone would have
   produced at that trial count.
3. **Fold consistency.** A result carried by one lucky fold is rejected even if
   the aggregate looks strong.
4. **Cost realism.** A candidate whose edge is smaller than its own fee drag is
   rejected regardless of Sharpe.
5. **Incumbent margin.** A challenger must beat the incumbent by a margin
   exceeding the incumbent's own measurement error, not merely by a point
   estimate.
6. **Live-shape agreement.** Out-of-sample drawdown must not exceed the risk
   profile's limit, or promotion would install a strategy the kill switch will
   immediately halt.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping, Sequence

from spintrader.backtest.engine import WalkForwardResult
from spintrader.backtest.scorecard import Scorecard, deflated_sharpe
from spintrader.core.types import utcnow

log = logging.getLogger(__name__)

ZERO = Decimal("0")


class Rejection(str, Enum):
    """Why a candidate was refused. Recorded, so patterns are visible."""
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_SIGNIFICANT = "not_significant"
    INCONSISTENT_FOLDS = "inconsistent_folds"
    COST_EXCEEDS_EDGE = "cost_exceeds_edge"
    NO_MARGIN_OVER_INCUMBENT = "no_margin_over_incumbent"
    DRAWDOWN_EXCEEDS_LIMIT = "drawdown_exceeds_limit"
    NEGATIVE_RETURN = "negative_return"


@dataclass(slots=True)
class PromotionVerdict:
    """The gate's ruling, with every check's outcome recorded."""
    candidate: str
    promoted: bool
    rejections: list[Rejection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    deflated_sharpe: float = 0.0
    n_trials: int = 1
    incumbent: str | None = None
    decided_at: datetime = field(default_factory=utcnow)

    def reject(self, reason: Rejection, note: str) -> "PromotionVerdict":
        self.promoted = False
        self.rejections.append(reason)
        self.notes.append(note)
        return self

    def summary(self) -> str:
        head = f"{self.candidate}: {'PROMOTED' if self.promoted else 'REJECTED'}"
        if self.incumbent:
            head += f" (vs {self.incumbent})"
        head += f" DSR={self.deflated_sharpe:.3f} trials={self.n_trials}"
        if self.notes:
            head += " | " + "; ".join(self.notes)
        return head


@dataclass
class TrialLedger:
    """Lifetime count of candidates evaluated, per objective.

    Kept per objective rather than globally: testing crypto strategies does not
    inflate the multiple-testing burden on an unrelated equity objective. But
    within an objective the count only ever grows, because that is the number
    of chances the loop had to find noise that looked like signal.
    """
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, objective: str, n: int = 1) -> int:
        self.counts[objective] = self.counts.get(objective, 0) + n
        return self.counts[objective]

    def trials(self, objective: str) -> int:
        return max(1, self.counts.get(objective, 0))

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(slots=True)
class PromotionPolicy:
    """Thresholds the gate applies. Deliberately strict."""
    min_observations: int = 250
    min_folds: int = 3
    # Fraction of folds that must be individually profitable. A result carried
    # by one fold is a lucky window, not a strategy.
    min_profitable_fold_ratio: float = 0.5
    min_deflated_sharpe: float = 0.95
    # Edge must exceed cost drag by this multiple. At 1.5, a strategy whose
    # gross edge is only 20% above its fees is refused -- the margin is inside
    # the error of the cost model itself.
    min_edge_to_cost: float = 1.5
    # Challenger must beat the incumbent's Sharpe by this many standard errors.
    min_incumbent_margin_sigmas: float = 1.0
    max_drawdown: Decimal = Decimal("0.25")


class PromotionGate:
    """Evaluates candidates against an incumbent and a policy."""

    def __init__(
        self,
        policy: PromotionPolicy | None = None,
        ledger: TrialLedger | None = None,
    ) -> None:
        self.policy = policy or PromotionPolicy()
        self.ledger = ledger or TrialLedger()
        self.history: list[PromotionVerdict] = []

    def evaluate(
        self,
        candidate: str,
        result: WalkForwardResult,
        objective: str,
        incumbent: Scorecard | None = None,
        incumbent_name: str | None = None,
        max_drawdown: Decimal | None = None,
    ) -> PromotionVerdict:
        """Decide whether ``candidate`` may replace the incumbent."""
        trials = self.ledger.record(objective)
        verdict = PromotionVerdict(
            candidate=candidate, promoted=True, n_trials=trials,
            incumbent=incumbent_name,
        )

        card = result.combined
        if card is None:
            self.history.append(
                verdict.reject(Rejection.INSUFFICIENT_DATA,
                               "walk-forward produced no combined scorecard")
            )
            return verdict

        # --- sample adequacy ---------------------------------------------
        if result.n_folds < self.policy.min_folds:
            verdict.reject(
                Rejection.INSUFFICIENT_DATA,
                f"{result.n_folds} folds is below the {self.policy.min_folds} minimum",
            )
        if card.n_observations < self.policy.min_observations:
            verdict.reject(
                Rejection.INSUFFICIENT_DATA,
                f"{card.n_observations} observations is below the "
                f"{self.policy.min_observations} minimum",
            )

        # --- direction ----------------------------------------------------
        if card.total_return <= 0:
            verdict.reject(
                Rejection.NEGATIVE_RETURN,
                f"out-of-sample return {card.total_return:.2%} is not positive",
            )

        # --- significance, adjusted for the loop's lifetime search --------
        dsr = deflated_sharpe(
            card.sharpe, card.n_observations, card.skew, card.excess_kurtosis,
            n_trials=trials, periods_per_year=card.periods_per_year,
        )
        verdict.deflated_sharpe = dsr
        if dsr < self.policy.min_deflated_sharpe:
            verdict.reject(
                Rejection.NOT_SIGNIFICANT,
                f"deflated Sharpe {dsr:.3f} below {self.policy.min_deflated_sharpe} "
                f"after {trials} lifetime trials (raw Sharpe {card.sharpe:.2f})",
            )

        # --- fold consistency ---------------------------------------------
        if result.folds:
            profitable = sum(
                1 for fold in result.folds
                if fold.scorecard and fold.scorecard.total_return > 0
            )
            ratio = profitable / len(result.folds)
            if ratio < self.policy.min_profitable_fold_ratio:
                verdict.reject(
                    Rejection.INCONSISTENT_FOLDS,
                    f"only {profitable}/{len(result.folds)} folds profitable, "
                    f"below the {self.policy.min_profitable_fold_ratio:.0%} floor",
                )

        # --- cost realism -------------------------------------------------
        # cost_drag is fees as a fraction of gross P&L. Above 1/min_edge_to_cost
        # the strategy is mostly paying the exchange.
        if card.cost_drag > 0:
            edge_to_cost = 1.0 / card.cost_drag if card.cost_drag > 0 else float("inf")
            if edge_to_cost < self.policy.min_edge_to_cost:
                verdict.reject(
                    Rejection.COST_EXCEEDS_EDGE,
                    f"edge is only {edge_to_cost:.2f}x fee drag "
                    f"({card.cost_drag:.1%} of gross P&L), below the "
                    f"{self.policy.min_edge_to_cost}x floor",
                )

        # --- drawdown compatible with the live kill switch ----------------
        limit = max_drawdown if max_drawdown is not None else self.policy.max_drawdown
        if abs(Decimal(str(card.max_drawdown))) > limit:
            verdict.reject(
                Rejection.DRAWDOWN_EXCEEDS_LIMIT,
                f"out-of-sample drawdown {card.max_drawdown:.1%} exceeds the "
                f"{limit:.1%} limit; the kill switch would halt this strategy",
            )

        # --- must beat the incumbent by more than measurement error --------
        if incumbent is not None:
            margin = card.sharpe - incumbent.sharpe
            required = self.policy.min_incumbent_margin_sigmas * max(
                incumbent.sharpe_stderr, card.sharpe_stderr
            )
            if margin < required:
                verdict.reject(
                    Rejection.NO_MARGIN_OVER_INCUMBENT,
                    f"Sharpe {card.sharpe:.2f} vs incumbent {incumbent.sharpe:.2f}: "
                    f"margin {margin:+.2f} is within measurement error "
                    f"(needs {required:+.2f})",
                )

        if verdict.promoted:
            verdict.notes.append(
                f"cleared all gates: Sharpe {card.sharpe:.2f}"
                f"±{card.sharpe_stderr:.2f}, DSR {dsr:.3f}, "
                f"{result.n_folds} folds, maxDD {card.max_drawdown:.1%}"
            )
            log.info("PROMOTED %s for %s: %s", candidate, objective, verdict.notes[-1])
        else:
            log.info("rejected %s for %s: %s", candidate, objective,
                     "; ".join(verdict.notes))

        self.history.append(verdict)
        return verdict

    # -- diagnostics -------------------------------------------------------

    def rejection_profile(self) -> dict[str, int]:
        """Counts by rejection reason across the gate's history.

        Useful as a diagnostic on the loop itself: if almost everything is
        rejected as NOT_SIGNIFICANT the candidate generator is producing noise,
        whereas a preponderance of COST_EXCEEDS_EDGE means it is finding real
        patterns too small to trade.
        """
        counts: dict[str, int] = {}
        for verdict in self.history:
            for reason in verdict.rejections:
                counts[reason.value] = counts.get(reason.value, 0) + 1
        return counts

    def promotion_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for v in self.history if v.promoted) / len(self.history)

    def health(self) -> dict[str, object]:
        rate = self.promotion_rate()
        warnings: list[str] = []
        if rate > 0.3 and len(self.history) >= 10:
            # A gate that approves a third of everything it sees is not
            # filtering; either the thresholds have drifted or the trial count
            # is being reset somewhere.
            warnings.append(
                f"promotion rate {rate:.0%} is implausibly high for a search "
                f"process; check that trial counts are cumulative"
            )
        if rate == 0.0 and len(self.history) >= 25:
            warnings.append(
                "nothing has been promoted in 25+ evaluations; the candidate "
                "generator may be producing nothing viable"
            )
        return {
            "evaluations": len(self.history),
            "promoted": sum(1 for v in self.history if v.promoted),
            "promotion_rate": rate,
            "rejections": self.rejection_profile(),
            "trials": self.ledger.as_dict(),
            "warnings": warnings,
        }


__all__ = [
    "PromotionGate", "PromotionPolicy", "PromotionVerdict", "Rejection",
    "TrialLedger",
]
