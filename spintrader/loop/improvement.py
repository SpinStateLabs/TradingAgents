"""The improvement cycle: generate candidates, prove them, promote the best.

This is the orchestrator that closes the self-improvement loop. The scoring half
already exists -- the walk-forward backtester, the :class:`PromotionGate`, the
:class:`TrialLedger`, the reliability tracker -- and refuses almost everything.
This module supplies the *generative* half and wires the two together, under one
invariant that the whole edifice depends on:

    **Every candidate that is evaluated is counted as a trial, before its result
    can influence anything.**

If the loop could generate a hundred candidates, quietly backtest them, and only
run the best one through the gate, the deflated-Sharpe correction would see a
trial count of one and wave through what is really a 1-in-100 fluke. So here:
``PromotionGate.evaluate`` (which increments the ledger) is called on *every*
fresh candidate, and the only candidates skipped are those the research memory
has already evaluated -- which were already counted. Skipping a re-test does not
under-count; re-running it would over-count.

A round evaluates each fresh candidate against a single fixed incumbent (the
current champion, re-backtested for an honest same-data comparison), then
promotes the highest-Sharpe challenger that cleared the gate. Evaluating against
one incumbent rather than a moving target keeps the round order-independent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Sequence

from spintrader.agents.personas.baseline_trend import BaselineTrendAgent
from spintrader.backtest.runner import run_backtest
from spintrader.core.config import Aggression, risk_profile
from spintrader.core.types import AssetClass, Bar
from spintrader.loop.promotion import PromotionGate, PromotionVerdict, Rejection
from spintrader.research.factory import CandidateConfig, CandidateFactory
from spintrader.research.memory import ResearchMemory, TrialRecord

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RoundResult:
    """What one improvement round did."""
    objective: str
    evaluated: int
    skipped: int
    promoted_key: str | None
    incumbent_key: str | None
    verdicts: list[PromotionVerdict] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.promoted_key is not None

    def summary(self) -> str:
        head = (
            f"{self.objective}: evaluated {self.evaluated}, skipped "
            f"{self.skipped} (already tried)"
        )
        if self.promoted_key:
            head += f" -> PROMOTED {self.promoted_key}"
            if self.incumbent_key:
                head += f" over {self.incumbent_key}"
        elif self.incumbent_key:
            head += f" -> no challenger beat {self.incumbent_key}"
        else:
            head += " -> nothing cleared the gate"
        return head


class ImprovementCycle:
    """Runs rounds of candidate generation, evaluation and promotion."""

    def __init__(
        self,
        gate: PromotionGate,
        memory: ResearchMemory,
        factory: CandidateFactory | None = None,
        strategy_cls: type = BaselineTrendAgent,
        backtest_fn: Callable[..., Any] = run_backtest,
    ) -> None:
        self.gate = gate
        self.memory = memory
        self.factory = factory or CandidateFactory()
        self.strategy_cls = strategy_cls
        self.backtest_fn = backtest_fn
        # Resolve a champion's family back to its strategy class when re-backtesting.
        self._family_cls = self.factory.family_classes()

    def run_round(
        self,
        objective: str,
        symbol: str,
        bars: Sequence[Bar],
        *,
        asset_class: AssetClass = AssetClass.CRYPTO,
        aggression: Aggression | str = Aggression.MODERATE,
        starting_cash: Decimal | str | float = "1000",
        n_candidates: int | None = None,
    ) -> RoundResult:
        """Evaluate fresh candidates and promote the best that clears the gate."""
        profile = risk_profile(aggression)
        max_dd = profile.max_drawdown_limit

        # 1. The incumbent: the champion so far, re-backtested on this data for an
        #    honest same-data Sharpe. This does NOT go through the gate, so it is
        #    not counted as a new trial -- it is the reference, not a candidate.
        champion = self.memory.best(objective)
        incumbent_card = None
        incumbent_key = None
        if champion is not None:
            champion_cls = self._family_cls.get(champion.family, self.strategy_cls)
            inc_run = self._backtest(champion.config, symbol, bars, asset_class,
                                     aggression, starting_cash, champion_cls)
            incumbent_card = self._card(inc_run)
            incumbent_key = champion.config_key

        # 2. Fresh candidates only -- the memory's keys are excluded, so nothing
        #    already evaluated (and counted) is re-tested.
        seen = self.memory.seen_keys(objective)
        skipped = sum(1 for c in self.factory.all_configs() if c.key in seen)
        candidates = self.factory.generate(n_candidates, avoid=seen)

        verdicts: list[PromotionVerdict] = []
        evaluated = 0
        promoted: list[tuple[PromotionVerdict, Any, CandidateConfig]] = []

        for cfg in candidates:
            run = self._backtest(cfg.to_dict(), symbol, bars, asset_class,
                                 aggression, starting_cash, cfg.strategy_cls)
            wf = getattr(run, "walk_forward", None)
            if wf is None or wf.combined is None:
                # Cannot evaluate (too few bars for a fold split). Record it as
                # seen so the round does not keep re-backtesting it, but do NOT
                # run the gate -- an aborted test is not a trial.
                self.memory.record(TrialRecord(
                    objective=objective, config_key=cfg.key, config=cfg.to_dict(),
                    promoted=False, deflated_sharpe=0.0, n_trials=0, family=cfg.family,
                    rejections=[Rejection.INSUFFICIENT_DATA.value],
                    note="no walk-forward result (too few bars)",
                ))
                continue

            verdict = self.gate.evaluate(
                cfg.name, wf, objective,
                incumbent=incumbent_card, incumbent_name=incumbent_key,
                max_drawdown=max_dd,
            )
            evaluated += 1
            card = wf.combined
            self.memory.record(TrialRecord(
                objective=objective, config_key=cfg.key, config=cfg.to_dict(),
                promoted=verdict.promoted, deflated_sharpe=verdict.deflated_sharpe,
                n_trials=verdict.n_trials, family=cfg.family,
                sharpe=card.sharpe, total_return=card.total_return,
                max_drawdown=card.max_drawdown,
                rejections=[r.value for r in verdict.rejections],
                note="; ".join(verdict.notes),
            ))
            verdicts.append(verdict)
            if verdict.promoted:
                promoted.append((verdict, card, cfg))

        # 3. The winner is the highest-Sharpe challenger that cleared the gate.
        winner = max(promoted, key=lambda t: t[1].sharpe) if promoted else None
        self.memory.save(objective)

        result = RoundResult(
            objective=objective, evaluated=evaluated, skipped=skipped,
            promoted_key=(winner[2].key if winner else None),
            incumbent_key=incumbent_key, verdicts=verdicts,
        )
        log.info("improvement round: %s", result.summary())
        return result

    # -- helpers -----------------------------------------------------------

    def _backtest(self, config, symbol, bars, asset_class, aggression, cash,
                  strategy_cls=None):
        return self.backtest_fn(
            strategy_cls or self.strategy_cls, symbol, bars,
            asset_class=asset_class, aggression=aggression,
            starting_cash=cash, walk_forward=True, n_trials=1,
            strategy_kwargs=dict(config),
        )

    @staticmethod
    def _card(run):
        wf = getattr(run, "walk_forward", None)
        if wf is not None and wf.combined is not None:
            return wf.combined
        return run.result.scorecard


__all__ = ["ImprovementCycle", "RoundResult"]
