"""Research memory: what the improvement loop has already tried, and learned.

Two jobs, both about not wasting the scarcest resource in a search process --
independent trials:

* **Don't pay twice.** Re-evaluating a configuration already tested burns compute
  for no new information. :meth:`ResearchMemory.seen` lets the orchestrator skip
  a config it has a record for, so a round only spends trials on genuinely new
  candidates.
* **Don't lose the trail.** Every evaluation -- promoted or rejected, and *why*
  rejected -- is recorded, so the search is auditable after the fact and the
  rejection profile can tell whether the generator is producing noise or real
  patterns too small to trade.

Recording here is deliberately separate from *counting* in the
:class:`~spintrader.loop.promotion.TrialLedger`. The ledger counts lifetime
trials for the multiple-testing correction; this memory records their content.
A config that was already evaluated was already counted, so skipping it via
:meth:`seen` does not under-count -- re-running it would *over*-count.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from spintrader.core.types import utcnow

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrialRecord:
    """The outcome of evaluating one candidate against the promotion gate."""
    objective: str
    config_key: str
    config: dict[str, Any]
    promoted: bool
    deflated_sharpe: float
    n_trials: int
    sharpe: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    rejections: list[str] = field(default_factory=list)
    note: str = ""
    ts: datetime = field(default_factory=utcnow)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "objective": self.objective,
            "config_key": self.config_key,
            "config": self.config,
            "promoted": self.promoted,
            "deflated_sharpe": self.deflated_sharpe,
            "n_trials": self.n_trials,
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "rejections": list(self.rejections),
            "note": self.note,
            "ts": self.ts.isoformat(),
        }
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrialRecord":
        ts = d.get("ts")
        return cls(
            objective=d["objective"], config_key=d["config_key"],
            config=dict(d.get("config", {})), promoted=bool(d["promoted"]),
            deflated_sharpe=float(d.get("deflated_sharpe", 0.0)),
            n_trials=int(d.get("n_trials", 1)),
            sharpe=d.get("sharpe"), total_return=d.get("total_return"),
            max_drawdown=d.get("max_drawdown"),
            rejections=list(d.get("rejections", [])), note=d.get("note", ""),
            ts=datetime.fromisoformat(ts) if isinstance(ts, str) else utcnow(),
        )


class ResearchMemory:
    """Per-objective record of every candidate the loop has evaluated.

    In-memory by default; pass a :class:`~spintrader.data.store.Store` to persist
    across runs via the research cache, so the loop resumes its search rather than
    restarting it (and re-counting trials it already paid for).
    """

    def __init__(self, store: Any | None = None) -> None:
        self._store = store
        # objective -> {config_key: TrialRecord}
        self._records: dict[str, dict[str, TrialRecord]] = {}

    # -- lookup ------------------------------------------------------------

    def seen(self, objective: str, config_key: str) -> bool:
        return config_key in self._records.get(objective, {})

    def seen_keys(self, objective: str) -> set[str]:
        return set(self._records.get(objective, {}))

    def records(self, objective: str) -> list[TrialRecord]:
        return list(self._records.get(objective, {}).values())

    def best(self, objective: str) -> TrialRecord | None:
        """The highest-Sharpe *promoted* record -- the current champion."""
        promoted = [
            r for r in self.records(objective)
            if r.promoted and r.sharpe is not None
        ]
        return max(promoted, key=lambda r: r.sharpe) if promoted else None

    # -- recording ---------------------------------------------------------

    def record(self, rec: TrialRecord) -> None:
        self._records.setdefault(rec.objective, {})[rec.config_key] = rec

    # -- persistence -------------------------------------------------------

    def load(self, objective: str) -> int:
        """Load persisted records for an objective. Returns how many were read."""
        if self._store is None:
            return 0
        payload = self._store.cache_get(self._cache_key(objective))
        if not payload:
            return 0
        rows = payload.get("records", []) if isinstance(payload, dict) else []
        bucket = self._records.setdefault(objective, {})
        for row in rows:
            rec = TrialRecord.from_dict(row)
            bucket[rec.config_key] = rec
        return len(rows)

    def save(self, objective: str) -> None:
        """Persist all records for an objective, if a store is configured."""
        if self._store is None:
            return
        payload = {"records": [r.as_dict() for r in self.records(objective)]}
        self._store.cache_put(
            self._cache_key(objective), source="research_memory", payload=payload,
        )

    @staticmethod
    def _cache_key(objective: str) -> str:
        return f"research_memory:{objective}"

    # -- diagnostics -------------------------------------------------------

    def rejection_profile(self, objective: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.records(objective):
            for reason in rec.rejections:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def summary(self, objective: str) -> dict[str, Any]:
        recs = self.records(objective)
        champion = self.best(objective)
        return {
            "objective": objective,
            "evaluated": len(recs),
            "promoted": sum(1 for r in recs if r.promoted),
            "champion": champion.config_key if champion else None,
            "champion_sharpe": champion.sharpe if champion else None,
            "rejections": self.rejection_profile(objective),
        }


__all__ = ["ResearchMemory", "TrialRecord"]
