"""Tests for the daily recalibration agent (task 16).

Offline, deterministic, no LLM and no network. Recalibration is a governance
action layered on top of the attribution scoring, and its whole value is that it
moves persona weights *slowly and only on evidence*. So almost every test here
pins a guardrail: weights move toward realised skill but never further than the
daily step, a thin sample is left untouched, the floor cannot be breached, and
every ruling is recorded old->new for the audit trail.

Skill is controlled precisely through the Brier scoring: a persona that is
confidently long and right scores skill +1 on each outcome; confidently long and
wrong scores -3; neutral scores 0. Shrinkage toward the 0.5 prior with
PRIOR_STRENGTH=30 pseudo-observations then makes the skill-implied target
predictable, e.g. 40 perfect outcomes shrink to skill 55/70 = 0.785714, which
maps to reliability 0.892857.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal

from spintrader.agents.personas.roster import SIMONS, TALEB, THORP
from spintrader.agents.personas.spec import PersonaRegistry
from spintrader.core.types import utcnow
from spintrader.loop.attribution import (
    MAX_STEP,
    MIN_RELIABILITY,
    PRIOR_STRENGTH,
    ForecastOutcome,
    ReliabilityTracker,
    skill_to_reliability,
)
from spintrader.loop.recalibration import (
    PersonaRecalibration,
    RecalibrationAgent,
    RecalibrationPolicy,
    RecalibrationReport,
    default_narrative,
    recalibrate,
)

D = Decimal
T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)

# 40 perfect outcomes shrink to 55/70 skill -> this reliability. Derived, not
# hand-tuned: it is what the attribution module will actually produce.
GOOD_TARGET = skill_to_reliability((40 * 1.0 + PRIOR_STRENGTH * 0.5) / (40 + PRIOR_STRENGTH))


def outcomes(persona, n, quality="good", weight="1.0"):
    """Build ``n`` scored outcomes of a chosen quality for one persona.

    good     -> confidently long and right, Brier skill +1 each
    bad      -> confidently long and wrong, Brier skill -3 each
    neutral  -> no directional conviction, Brier skill 0 each
    """
    if quality == "good":
        direction, ret = D("1.0"), D("0.05")
    elif quality == "bad":
        direction, ret = D("1.0"), D("-0.05")
    elif quality == "neutral":
        direction, ret = D("0"), D("0.05")
    else:  # pragma: no cover - guard against a typo in a test
        raise ValueError(quality)
    return [
        ForecastOutcome(
            decision_id=f"{persona}-{i}", persona_key=persona,
            instrument_key="kraken:BTC-USD", direction=direction,
            weight=D(weight), realised_return=ret, resolved_at=T0,
        )
        for i in range(n)
    ]


def registry_with(**reliabilities):
    """A small registry of real specs, each set to a chosen reliability."""
    specs = {"simons": SIMONS, "thorp": THORP, "taleb": TALEB}
    reg = PersonaRegistry([specs[k] for k in reliabilities])
    for key, rel in reliabilities.items():
        reg.update_reliability(key, D(rel))
    return reg


def tracker_with(*outcome_lists):
    tr = ReliabilityTracker()
    for lst in outcome_lists:
        tr.record_many(lst)
    return tr


# --------------------------------------------------------------------------
# Movement toward skill, within the step bound
# --------------------------------------------------------------------------

class MovementTests(unittest.TestCase):
    def test_derived_target_is_what_we_expect(self):
        # Sanity-check the fixture: the reliability 40 perfect calls imply.
        self.assertEqual(GOOD_TARGET, D("0.892857"))

    def test_reliability_moves_toward_skill(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        self.assertTrue(entry.changed)
        # Target is ~0.89, but a single day may move at most MAX_STEP toward it.
        self.assertEqual(entry.new_reliability, D("0.5") + MAX_STEP)
        self.assertGreater(entry.new_reliability, entry.old_reliability)
        self.assertEqual(reg.get("simons").reliability, entry.new_reliability)

    def test_move_never_exceeds_the_step_bound(self):
        # Whatever the skill gap, one recalibration moves at most MAX_STEP.
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        self.assertTrue(entry.step_capped)
        self.assertLessEqual(abs(entry.delta), MAX_STEP)
        self.assertEqual(abs(entry.delta), MAX_STEP)  # gap exceeds the cap here

    def test_good_persona_gains_and_bad_persona_loses(self):
        reg = registry_with(simons="0.5", thorp="0.8")
        report = recalibrate(reg, tracker_with(
            outcomes("simons", 40, "good"),   # skill +1
            outcomes("thorp", 40, "bad"),     # skill -3
        ))
        good, bad = report.get("simons"), report.get("thorp")
        self.assertGreater(good.new_reliability, good.old_reliability)   # gained
        self.assertLess(bad.new_reliability, bad.old_reliability)        # lost
        self.assertEqual(good.new_reliability, D("0.6"))
        self.assertEqual(bad.new_reliability, D("0.7"))
        # And the registry the panel reads has actually been updated.
        self.assertEqual(reg.get("simons").reliability, D("0.6"))
        self.assertEqual(reg.get("thorp").reliability, D("0.7"))

    def test_default_prior_of_one_is_corrected_downward(self):
        # A persona left at the untested prior of 1.0 is pulled toward its
        # (high, but sub-1) realised skill target, not left overconfident.
        reg = registry_with(simons="1.0")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        self.assertEqual(entry.new_reliability, D("1.0") - MAX_STEP)
        self.assertLess(entry.new_reliability, entry.old_reliability)

    def test_converges_over_several_days_without_jumping(self):
        # Repeated daily recalibration against an accumulating tracker walks the
        # weight to the target in steps, never in a jump.
        reg = registry_with(simons="0.5")
        tr = tracker_with(outcomes("simons", 40, "good"))
        last = reg.get("simons").reliability
        for _ in range(8):
            entry = recalibrate(reg, tr).get("simons")
            self.assertLessEqual(abs(entry.new_reliability - last), MAX_STEP)
            last = entry.new_reliability
        self.assertEqual(reg.get("simons").reliability, GOOD_TARGET)


# --------------------------------------------------------------------------
# Evidence gate: thin samples are untouched
# --------------------------------------------------------------------------

class EvidenceGateTests(unittest.TestCase):
    def test_low_evidence_persona_is_left_unchanged(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 10, "good")))
        entry = report.get("simons")
        self.assertFalse(entry.changed)
        self.assertEqual(entry.new_reliability, entry.old_reliability)
        self.assertEqual(entry.new_reliability, D("0.5"))
        self.assertEqual(entry.n_observations, 10)
        self.assertIn("evidence threshold", entry.reason)
        # Registry is untouched.
        self.assertEqual(reg.get("simons").reliability, D("0.5"))

    def test_exactly_at_threshold_is_allowed_to_move(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(
            reg, tracker_with(outcomes("simons", PRIOR_STRENGTH, "good")))
        self.assertTrue(report.get("simons").changed)

    def test_secondary_evidence_weight_gate(self):
        # A stricter evidence-weight bar can hold back a persona that clears the
        # raw observation count.
        reg = registry_with(simons="0.5")
        policy = RecalibrationPolicy(min_observations=1, min_evidence_weight=0.9)
        report = recalibrate(
            reg, tracker_with(outcomes("simons", 40, "good")), policy=policy)
        entry = report.get("simons")
        self.assertFalse(entry.changed)          # 40/70 evidence < 0.9
        self.assertIn("evidence weight", entry.reason)

    def test_persona_with_no_outcomes_is_reported_but_untouched(self):
        reg = registry_with(simons="0.5", taleb="0.9")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        self.assertIn("taleb", report.skipped_no_data)
        self.assertIsNone(report.get("taleb"))
        self.assertEqual(reg.get("taleb").reliability, D("0.9"))


# --------------------------------------------------------------------------
# Floor / clamp
# --------------------------------------------------------------------------

class ClampTests(unittest.TestCase):
    def test_floor_holds_against_a_terrible_run(self):
        reg = registry_with(simons="0.12")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "bad")))
        entry = report.get("simons")
        # Skill target is 0.0; stepping down from 0.12 by the cap reaches 0.02,
        # which the floor lifts back to 0.05.
        self.assertTrue(entry.step_capped)
        self.assertTrue(entry.floored)
        self.assertEqual(entry.new_reliability, MIN_RELIABILITY)
        self.assertGreaterEqual(entry.new_reliability, MIN_RELIABILITY)

    def test_reliability_never_falls_below_the_floor(self):
        # Even starting at the floor with the worst possible skill, it holds.
        reg = registry_with(simons=str(MIN_RELIABILITY))
        report = recalibrate(reg, tracker_with(outcomes("simons", 60, "bad")))
        entry = report.get("simons")
        self.assertEqual(entry.new_reliability, MIN_RELIABILITY)
        self.assertGreaterEqual(entry.new_reliability, MIN_RELIABILITY)

    def test_custom_floor_is_respected(self):
        reg = registry_with(simons="0.30")
        policy = RecalibrationPolicy(floor=D("0.25"), max_step=D("0.50"))
        report = recalibrate(
            reg, tracker_with(outcomes("simons", 40, "bad")), policy=policy)
        entry = report.get("simons")
        self.assertEqual(entry.new_reliability, D("0.25"))
        self.assertTrue(entry.floored)


# --------------------------------------------------------------------------
# The report: old/new recorded correctly for the audit trail
# --------------------------------------------------------------------------

class ReportTests(unittest.TestCase):
    def test_report_records_old_and_new(self):
        reg = registry_with(simons="0.5")
        before = reg.get("simons").reliability
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        after = reg.get("simons").reliability
        self.assertEqual(entry.old_reliability, before)
        self.assertEqual(entry.new_reliability, after)
        self.assertEqual(entry.delta, after - before)

    def test_unchanged_persona_reports_reason(self):
        # Already at the skill-implied target -> gate passes, nothing to do.
        reg = registry_with(simons=str(GOOD_TARGET))
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        self.assertFalse(entry.changed)
        self.assertEqual(entry.new_reliability, GOOD_TARGET)
        self.assertIn("already at", entry.reason)

    def test_report_carries_skill_and_evidence(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        entry = report.get("simons")
        self.assertEqual(entry.n_observations, 40)
        self.assertAlmostEqual(entry.raw_skill, 1.0)
        self.assertAlmostEqual(entry.shrunk_skill, 55 / 70)
        self.assertEqual(entry.skill_target, GOOD_TARGET)
        self.assertAlmostEqual(entry.evidence_weight, 40 / 70)

    def test_changed_and_unchanged_partition(self):
        reg = registry_with(simons="0.5", thorp="0.5")
        report = recalibrate(reg, tracker_with(
            outcomes("simons", 40, "good"),   # moves
            outcomes("thorp", 10, "good"),    # too little evidence
        ))
        self.assertEqual({e.persona_key for e in report.changed}, {"simons"})
        self.assertEqual({e.persona_key for e in report.unchanged}, {"thorp"})

    def test_report_as_dict_is_serialisable(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        d = report.as_dict()
        self.assertEqual(d["entries"][0]["persona"], "simons")
        self.assertEqual(d["entries"][0]["old_reliability"], "0.5")
        self.assertEqual(D(d["entries"][0]["new_reliability"]), D("0.6"))
        # Everything JSON-friendly (Decimals rendered as strings).
        import json
        json.dumps(d)

    def test_summary_strings_render(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        self.assertIn("recalibration", report.summary())
        self.assertIn("simons", report.get("simons").summary())


# --------------------------------------------------------------------------
# Unknown personas and functional entry point
# --------------------------------------------------------------------------

class WiringTests(unittest.TestCase):
    def test_tracked_persona_not_in_registry_is_flagged(self):
        reg = registry_with(simons="0.5")
        tr = tracker_with(
            outcomes("simons", 40, "good"),
            outcomes("ghost", 40, "good"),   # not in the registry
        )
        report = recalibrate(reg, tr)
        self.assertIn("ghost", report.skipped_unknown)
        self.assertIsNone(report.get("ghost"))

    def test_recalibrate_from_a_raw_outcome_batch(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, outcomes=outcomes("simons", 40, "good"))
        self.assertEqual(reg.get("simons").reliability, D("0.6"))
        self.assertTrue(report.get("simons").changed)

    def test_requires_a_tracker_or_outcomes(self):
        reg = registry_with(simons="0.5")
        with self.assertRaises(ValueError):
            recalibrate(reg)

    def test_rejects_both_tracker_and_outcomes(self):
        reg = registry_with(simons="0.5")
        with self.assertRaises(ValueError):
            recalibrate(reg, tracker_with(outcomes("simons", 40, "good")),
                        outcomes=outcomes("simons", 40, "good"))

    def test_window_limits_to_recent_outcomes(self):
        # Old good outcomes then a run of bad ones; a short window sees only the
        # recent bad, and the weight should be pulled down rather than up.
        reg = registry_with(simons="0.5")
        tr = tracker_with(
            outcomes("simons", 40, "good"),
            outcomes("simons", 40, "bad"),
        )
        report = recalibrate(reg, tr, window=40)
        # The last 40 are all bad -> target is low -> weight drops.
        self.assertLess(report.get("simons").new_reliability, D("0.5"))


# --------------------------------------------------------------------------
# Determinism and the optional narrator
# --------------------------------------------------------------------------

class DeterminismAndNarratorTests(unittest.TestCase):
    def test_recalibration_is_deterministic(self):
        def run_once():
            reg = registry_with(simons="0.5", thorp="0.8")
            rep = recalibrate(reg, tracker_with(
                outcomes("simons", 40, "good"),
                outcomes("thorp", 40, "bad"),
            ), now=T0)
            return {e.persona_key: e.new_reliability for e in rep.entries}
        self.assertEqual(run_once(), run_once())

    def test_injected_narrator_is_called(self):
        seen = {}

        def narrator(report):
            seen["report"] = report
            return "custom narrative"

        reg = registry_with(simons="0.5")
        report = recalibrate(
            reg, tracker_with(outcomes("simons", 40, "good")), narrator=narrator)
        self.assertEqual(report.narrative, "custom narrative")
        self.assertIs(seen["report"], report)

    def test_default_narrative_is_offline_and_mentions_movers(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(
            reg, tracker_with(outcomes("simons", 40, "good")),
            narrator=default_narrative)
        self.assertIn("simons", report.narrative)

    def test_default_narrative_when_nothing_moved(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(
            reg, tracker_with(outcomes("simons", 10, "good")),
            narrator=default_narrative)
        self.assertIn("No persona weights moved", report.narrative)

    def test_no_narrator_leaves_narrative_none(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        self.assertIsNone(report.narrative)


# --------------------------------------------------------------------------
# Policy validation and agent reuse
# --------------------------------------------------------------------------

class PolicyTests(unittest.TestCase):
    def test_invalid_policy_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            RecalibrationPolicy(max_step=D("0"))
        with self.assertRaises(ValueError):
            RecalibrationPolicy(floor=D("0.9"), ceiling=D("0.5"))
        with self.assertRaises(ValueError):
            RecalibrationPolicy(min_observations=-1)

    def test_agent_can_be_reused_across_registries(self):
        agent = RecalibrationAgent()
        for _ in range(2):
            reg = registry_with(simons="0.5")
            agent.run(reg, tracker_with(outcomes("simons", 40, "good")), now=T0)
            self.assertEqual(reg.get("simons").reliability, D("0.6"))

    def test_now_defaults_to_utcnow(self):
        reg = registry_with(simons="0.5")
        report = recalibrate(reg, tracker_with(outcomes("simons", 40, "good")))
        self.assertGreaterEqual(utcnow(), report.generated_at)
        self.assertEqual(report.generated_at.tzinfo, timezone.utc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
