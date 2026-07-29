"""Tests for the self-improvement loop.

This is the component most able to destroy the account, because its failure mode
is not a crash but confident, well-documented promotion of noise. Almost every
test here pins a refusal rather than a capability.

The two properties that matter:

* **Reliability moves slowly and on evidence.** Weights must not swing on ten
  observations, and oscillation must be visible as instability rather than pass
  for responsiveness.
* **The gate says no.** Cumulative trial accounting, fold consistency, cost
  realism and an incumbent margin exceeding measurement error each close one
  route by which a lucky backtest reaches live capital.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.backtest.engine import BacktestResult, WalkForwardResult
from spintrader.backtest.scorecard import Scorecard, sharpe_standard_error
from spintrader.core.types import Action
from spintrader.loop.attribution import (
    MAX_STEP, PRIOR_STRENGTH, ForecastOutcome, ReliabilityTracker,
    attribute_pnl, brier_score, brier_skill, direction_to_probability,
    resolve_decision, skill_to_reliability,
)
from spintrader.loop.promotion import (
    PromotionGate, PromotionPolicy, Rejection, TrialLedger,
)

D = Decimal
T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


def outcome(persona="simons", direction="0.8", weight="1.0", ret="0.05",
            decision_id="d1", pnl="0"):
    return ForecastOutcome(
        decision_id=decision_id, persona_key=persona,
        instrument_key="kraken:BTC-USD", direction=D(direction),
        weight=D(weight), realised_return=D(ret), resolved_at=T0,
        pnl_attributed=D(pnl),
    )


def card(sharpe=1.5, n=500, dd=-0.10, cost_drag=0.05, total_return=0.20,
         periods=365, trials=1):
    return Scorecard(
        n_observations=n, periods_per_year=periods, sharpe=sharpe,
        sharpe_stderr=sharpe_standard_error(sharpe, n, periods),
        max_drawdown=dd, cost_drag=cost_drag, total_return=total_return,
        n_trials=trials,
    )


def wf_result(combined, n_folds=4, profitable=4):
    folds = []
    for i in range(n_folds):
        fold = BacktestResult(
            strategy="cand", instrument_key="kraken:BTC-USD",
            start=T0, end=T0 + timedelta(days=30),
        )
        fold.scorecard = card(total_return=0.05 if i < profitable else -0.05)
        folds.append(fold)
    out = WalkForwardResult(strategy="cand", instrument_key="kraken:BTC-USD",
                            folds=folds)
    out.combined = combined
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

class ScoringTests(unittest.TestCase):
    def test_direction_maps_to_probability(self):
        self.assertEqual(direction_to_probability(D("1")), 1.0)
        self.assertEqual(direction_to_probability(D("-1")), 0.0)
        self.assertEqual(direction_to_probability(D("0")), 0.5)

    def test_perfect_call_scores_one(self):
        self.assertAlmostEqual(brier_skill(1.0, True), 1.0)

    def test_uninformative_call_scores_zero(self):
        # The reference point: a 50/50 forecast adds nothing and is scored as
        # adding nothing.
        self.assertAlmostEqual(brier_skill(0.5, True), 0.0)
        self.assertAlmostEqual(brier_skill(0.5, False), 0.0)

    def test_confidently_wrong_is_penalised_more_than_quietly_right_is_rewarded(self):
        # Deliberate asymmetry: loud errors should cost more than quiet
        # successes earn.
        loudly_wrong = brier_skill(1.0, False)
        quietly_right = brier_skill(0.6, True)
        self.assertLess(loudly_wrong, -1.0)
        self.assertGreater(quietly_right, 0.0)
        self.assertGreater(abs(loudly_wrong), abs(quietly_right))

    def test_brier_is_a_proper_scoring_rule(self):
        """Exaggerating confidence must not improve the expected score.

        This is why Brier is used rather than a hit rate. If a persona whose
        true belief is 0.7 scores better by claiming 1.0, the whole reliability
        signal becomes a measure of bravado.
        """
        true_p = 0.7
        honest = true_p * brier_score(0.7, True) + (1 - true_p) * brier_score(0.7, False)
        exaggerated = true_p * brier_score(1.0, True) + (1 - true_p) * brier_score(1.0, False)
        self.assertLess(honest, exaggerated)

    def test_skill_maps_to_reliability_with_neutral_midpoint(self):
        # An uninformative persona keeps a voice: its disagreement is
        # informative even when its direction is not.
        self.assertEqual(skill_to_reliability(0.0), D("0.5"))
        self.assertEqual(skill_to_reliability(1.0), D("1"))
        self.assertEqual(skill_to_reliability(-1.0), D("0"))

    def test_outcome_direction_correctness(self):
        self.assertTrue(outcome(direction="0.8", ret="0.05").was_directionally_right)
        self.assertFalse(outcome(direction="0.8", ret="-0.05").was_directionally_right)
        self.assertTrue(outcome(direction="-0.8", ret="-0.05").was_directionally_right)

    def test_abstention_is_never_counted_as_correct(self):
        self.assertFalse(outcome(direction="0", ret="0.05").was_directionally_right)


# --------------------------------------------------------------------------
# P&L attribution
# --------------------------------------------------------------------------

class AttributionTests(unittest.TestCase):
    def test_aligned_votes_share_the_profit(self):
        split = attribute_pnl(D("100"), [("a", D("0.8"), D("1")),
                                         ("b", D("0.4"), D("1"))])
        self.assertGreater(split["a"], split["b"])
        self.assertAlmostEqual(float(split["a"] + split["b"]), 100.0, places=6)

    def test_only_advocates_own_the_outcome(self):
        """Dissenters get nothing, not a debit.

        A persona that argued against the position and was overruled did not
        cause the trade. Debiting it for a loss it warned about would be
        perverse; its correctness is captured by the Brier skill score instead.
        """
        split = attribute_pnl(D("100"), [("bull", D("0.8"), D("1")),
                                         ("bear", D("-0.4"), D("1"))])
        self.assertGreater(split["bull"], D("0"))
        self.assertNotIn("bear", split)

    def test_dissenter_is_not_debited_on_a_losing_trade(self):
        # The regression this replaced: inferring market direction from the P&L
        # sign debited the bear who had correctly warned about the long.
        split = attribute_pnl(D("-100"), [("bull", D("0.8"), D("1")),
                                          ("bear", D("-0.4"), D("1"))])
        self.assertEqual(split["bull"], D("-100"))
        self.assertNotIn("bear", split)

    def test_attributions_sum_to_the_realised_pnl(self):
        # The ledger has to reconcile.
        for pnl in (D("100"), D("-100")):
            with self.subTest(pnl=pnl):
                split = attribute_pnl(pnl, [("a", D("0.8"), D("1")),
                                            ("b", D("0.4"), D("1")),
                                            ("c", D("-0.9"), D("1"))])
                self.assertEqual(sum(split.values()), pnl)

    def test_position_direction_comes_from_the_votes_not_the_pnl(self):
        # Net short position that happened to lose: the shorts own it.
        split = attribute_pnl(D("-100"), [("bear1", D("-0.8"), D("1")),
                                          ("bear2", D("-0.6"), D("1")),
                                          ("bull", D("0.2"), D("1"))])
        self.assertIn("bear1", split)
        self.assertIn("bear2", split)
        self.assertNotIn("bull", split)

    def test_perfectly_split_panel_attributes_to_nobody(self):
        # No net position was advocated, so nobody owns the outcome.
        split = attribute_pnl(D("100"), [("bull", D("0.5"), D("1")),
                                         ("bear", D("-0.5"), D("1"))])
        self.assertEqual(split, {})

    def test_weight_scales_the_share(self):
        split = attribute_pnl(D("100"), [("heavy", D("0.5"), D("1.0")),
                                         ("light", D("0.5"), D("0.2"))])
        self.assertGreater(split["heavy"], split["light"])

    def test_abstentions_receive_nothing(self):
        split = attribute_pnl(D("100"), [("voter", D("0.8"), D("1")),
                                         ("abstainer", D("0"), D("1"))])
        self.assertNotIn("abstainer", split)

    def test_no_votes_yields_no_attribution(self):
        self.assertEqual(attribute_pnl(D("100"), []), {})


class ResolveDecisionTests(unittest.TestCase):
    CONTRIBUTIONS = {
        "votes": [
            {"persona": "simons", "action": "buy", "confidence": "0.8",
             "weight": "1.0"},
            {"persona": "taleb", "action": "sell", "confidence": "0.6",
             "weight": "0.5"},
            {"persona": "berkshire", "action": "hold", "confidence": "0",
             "weight": "1.0"},
        ],
    }

    def test_reads_back_the_vote_record(self):
        outcomes = resolve_decision(
            "d1", "kraken:BTC-USD", self.CONTRIBUTIONS,
            entry_price=D("100"), exit_price=D("110"), resolved_at=T0,
            realised_pnl=D("50"),
        )
        keys = {o.persona_key for o in outcomes}
        self.assertEqual(keys, {"simons", "taleb"})     # holder excluded

    def test_realised_return_computed_from_prices(self):
        outcomes = resolve_decision(
            "d1", "kraken:BTC-USD", self.CONTRIBUTIONS,
            entry_price=D("100"), exit_price=D("110"), resolved_at=T0,
        )
        self.assertEqual(outcomes[0].realised_return, D("0.1"))

    def test_sell_vote_becomes_negative_direction(self):
        outcomes = resolve_decision(
            "d1", "kraken:BTC-USD", self.CONTRIBUTIONS,
            entry_price=D("100"), exit_price=D("110"), resolved_at=T0,
        )
        taleb = next(o for o in outcomes if o.persona_key == "taleb")
        self.assertLess(taleb.direction, D("0"))

    def test_malformed_vote_is_skipped_not_fatal(self):
        outcomes = resolve_decision(
            "d1", "kraken:BTC-USD",
            {"votes": [{"persona": "x"}, {"persona": "simons", "action": "buy",
                                          "confidence": "0.8", "weight": "1.0"}]},
            entry_price=D("100"), exit_price=D("110"), resolved_at=T0,
        )
        self.assertEqual(len(outcomes), 1)

    def test_invalid_entry_price_raises(self):
        with self.assertRaises(ValueError):
            resolve_decision("d1", "k", self.CONTRIBUTIONS,
                             entry_price=D("0"), exit_price=D("110"),
                             resolved_at=T0)


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------

class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ReliabilityTracker()

    def test_no_data_keeps_the_current_weight(self):
        est = self.tracker.estimate("simons", current=D("0.8"))
        self.assertEqual(est.reliability, D("0.8"))
        self.assertEqual(est.n_observations, 0)

    def test_small_samples_barely_move_the_estimate(self):
        """The central guard against fitting noise.

        Ten correct calls is consistent with a good persona or a lucky one.
        Shrinkage must keep the estimate close to the prior.
        """
        for i in range(10):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="1.0",
                                        ret="0.05"))
        est = self.tracker.estimate("simons", current=D("0.5"))
        self.assertLess(est.evidence_weight, 0.3)
        self.assertFalse(est.is_established)
        self.assertLess(abs(est.reliability - D("0.5")), MAX_STEP + D("0.001"))

    def test_sustained_evidence_does_move_the_estimate(self):
        for i in range(120):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="1.0",
                                        ret="0.05"))
        est = self.tracker.estimate("simons", current=D("0.5"))
        self.assertTrue(est.is_established)
        self.assertGreater(est.evidence_weight, 0.7)
        self.assertGreater(est.raw_skill, 0.9)

    def test_per_update_movement_is_capped(self):
        for i in range(200):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="1.0",
                                        ret="-0.05"))     # always wrong
        est = self.tracker.estimate("simons", current=D("1.0"))
        self.assertTrue(est.clamped)
        self.assertGreaterEqual(est.reliability, D("1.0") - MAX_STEP)

    def test_reliability_never_reaches_zero_automatically(self):
        # A persona may be the only one applicable to a decision; permanent
        # silencing is a human's call.
        for i in range(500):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="1.0",
                                        ret="-0.05"))
        reliability = D("1.0")
        for _ in range(50):
            reliability = self.tracker.estimate("simons", current=reliability).reliability
        self.assertGreater(reliability, D("0"))

    def test_wrong_persona_loses_reliability_over_time(self):
        for i in range(200):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="1.0",
                                        ret="-0.05"))
        reliability = D("1.0")
        for _ in range(10):
            reliability = self.tracker.estimate("simons", current=reliability).reliability
        self.assertLess(reliability, D("0.6"))

    def test_hit_rate_reported(self):
        for i in range(10):
            self.tracker.record(outcome(decision_id=f"d{i}", direction="0.8",
                                        ret="0.05" if i < 7 else "-0.05"))
        est = self.tracker.estimate("simons")
        self.assertAlmostEqual(est.hit_rate, 0.7)

    def test_window_limits_the_estimate_to_recent_history(self):
        # A methodology's edge can decay; an all-history mean hides that.
        for i in range(60):
            self.tracker.record(outcome(decision_id=f"old{i}", direction="1.0",
                                        ret="0.05"))
        for i in range(60):
            self.tracker.record(outcome(decision_id=f"new{i}", direction="1.0",
                                        ret="-0.05"))
        all_history = self.tracker.estimate("simons", window=None)
        recent = self.tracker.estimate("simons", window=60)
        self.assertGreater(all_history.raw_skill, recent.raw_skill)

    def test_pnl_is_accumulated_separately_from_skill(self):
        self.tracker.record(outcome(direction="0.8", ret="0.05", pnl="10"))
        self.tracker.record(outcome(direction="0.8", ret="0.05", pnl="-3",
                                    decision_id="d2"))
        est = self.tracker.estimate("simons")
        self.assertEqual(est.pnl_attributed, D("7"))

    def test_estimate_all_covers_every_tracked_persona(self):
        self.tracker.record(outcome(persona="simons"))
        self.tracker.record(outcome(persona="taleb"))
        self.assertEqual(set(self.tracker.estimate_all()), {"simons", "taleb"})


class StabilityTests(unittest.TestCase):
    """Oscillation is fitting noise, not adapting."""

    def test_steady_estimates_report_high_stability(self):
        tracker = ReliabilityTracker()
        for i in range(200):
            tracker.record(outcome(decision_id=f"d{i}", direction="0.8", ret="0.05"))
        reliability = D("0.9")
        for _ in range(8):
            reliability = tracker.estimate("simons", current=reliability).reliability
        self.assertGreater(tracker.stability("simons"), 0.5)

    def test_alternating_evidence_is_reported_as_unstable(self):
        tracker = ReliabilityTracker(prior_strength=1, max_step=D("1"))
        reliability = D("0.5")
        for round_index in range(8):
            # Flip the persona between perfect and useless each round.
            good = round_index % 2 == 0
            for i in range(40):
                tracker.record(outcome(
                    decision_id=f"r{round_index}d{i}", direction="1.0",
                    ret="0.05" if good else "-0.05",
                ))
            reliability = tracker.estimate("simons", current=reliability).reliability
        self.assertLess(tracker.stability("simons"), 0.95)

    def test_insufficient_history_defaults_to_stable(self):
        tracker = ReliabilityTracker()
        self.assertEqual(tracker.stability("unknown"), 1.0)

    def test_health_lists_established_and_provisional(self):
        tracker = ReliabilityTracker()
        for i in range(PRIOR_STRENGTH + 5):
            tracker.record(outcome(persona="veteran", decision_id=f"v{i}"))
        tracker.record(outcome(persona="rookie"))
        tracker.estimate_all()
        health = tracker.health()
        self.assertIn("veteran", health["established"])
        self.assertIn("rookie", health["provisional"])


# --------------------------------------------------------------------------
# Promotion gate
# --------------------------------------------------------------------------

class TrialLedgerTests(unittest.TestCase):
    def test_counts_accumulate(self):
        ledger = TrialLedger()
        ledger.record("crypto")
        ledger.record("crypto")
        self.assertEqual(ledger.trials("crypto"), 2)

    def test_objectives_are_independent(self):
        ledger = TrialLedger()
        ledger.record("crypto", 50)
        self.assertEqual(ledger.trials("equity"), 1)

    def test_unseen_objective_reports_one_not_zero(self):
        # A zero trial count would disable the multiple-testing correction.
        self.assertEqual(TrialLedger().trials("new"), 1)


class PromotionGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = PromotionGate()

    def test_strong_candidate_is_promoted(self):
        verdict = self.gate.evaluate(
            "good", wf_result(card(sharpe=3.0, n=1000, dd=-0.08, cost_drag=0.05)),
            objective="crypto",
        )
        self.assertTrue(verdict.promoted, verdict.summary())

    def test_mined_noise_is_rejected(self):
        """The scenario this gate exists for.

        A Sharpe of 0.9 looks respectable and is exactly what 100 coin-flip
        variants produce on this data.
        """
        gate = PromotionGate()
        gate.ledger.record("crypto", 99)
        verdict = gate.evaluate("lucky", wf_result(card(sharpe=0.9, n=500)),
                                objective="crypto")
        self.assertFalse(verdict.promoted)
        self.assertIn(Rejection.NOT_SIGNIFICANT, verdict.rejections)

    def test_trial_count_is_cumulative_across_rounds(self):
        # Resetting per round would let 100 rounds of 10 pass as 10 tests.
        gate = PromotionGate()
        for i in range(30):
            gate.evaluate(f"c{i}", wf_result(card(sharpe=1.2, n=500)),
                          objective="crypto")
        self.assertEqual(gate.ledger.trials("crypto"), 30)
        later = gate.evaluate("c30", wf_result(card(sharpe=1.2, n=500)),
                              objective="crypto")
        first_dsr = gate.history[0].deflated_sharpe
        self.assertLess(later.deflated_sharpe, first_dsr)

    def test_negative_return_rejected(self):
        verdict = self.gate.evaluate(
            "loser", wf_result(card(sharpe=3.0, total_return=-0.05)),
            objective="crypto",
        )
        self.assertIn(Rejection.NEGATIVE_RETURN, verdict.rejections)

    def test_too_few_folds_rejected(self):
        verdict = self.gate.evaluate(
            "thin", wf_result(card(sharpe=3.0), n_folds=2, profitable=2),
            objective="crypto",
        )
        self.assertIn(Rejection.INSUFFICIENT_DATA, verdict.rejections)

    def test_inconsistent_folds_rejected(self):
        # Carried by one lucky window rather than a real edge.
        verdict = self.gate.evaluate(
            "lucky_fold", wf_result(card(sharpe=3.0), n_folds=4, profitable=1),
            objective="crypto",
        )
        self.assertIn(Rejection.INCONSISTENT_FOLDS, verdict.rejections)

    def test_cost_dominated_edge_rejected(self):
        # A real pattern too small to trade is still not tradeable.
        verdict = self.gate.evaluate(
            "expensive", wf_result(card(sharpe=3.0, cost_drag=0.8)),
            objective="crypto",
        )
        self.assertIn(Rejection.COST_EXCEEDS_EDGE, verdict.rejections)

    def test_excessive_drawdown_rejected(self):
        # Promoting this would install a strategy the kill switch halts at once.
        verdict = self.gate.evaluate(
            "volatile", wf_result(card(sharpe=3.0, dd=-0.45)),
            objective="crypto",
        )
        self.assertIn(Rejection.DRAWDOWN_EXCEEDS_LIMIT, verdict.rejections)

    def test_marginal_improvement_over_incumbent_rejected(self):
        incumbent = card(sharpe=1.40, n=1000)
        verdict = self.gate.evaluate(
            "barely_better", wf_result(card(sharpe=1.45, n=1000)),
            objective="crypto", incumbent=incumbent, incumbent_name="current",
        )
        self.assertIn(Rejection.NO_MARGIN_OVER_INCUMBENT, verdict.rejections)

    def test_clear_improvement_over_incumbent_accepted(self):
        incumbent = card(sharpe=1.0, n=1000)
        verdict = self.gate.evaluate(
            "much_better", wf_result(card(sharpe=3.5, n=1000, dd=-0.08)),
            objective="crypto", incumbent=incumbent, incumbent_name="current",
        )
        self.assertTrue(verdict.promoted, verdict.summary())

    def test_missing_scorecard_rejected(self):
        empty = WalkForwardResult(strategy="c", instrument_key="k")
        verdict = self.gate.evaluate("nodata", empty, objective="crypto")
        self.assertIn(Rejection.INSUFFICIENT_DATA, verdict.rejections)

    def test_rejections_accumulate_rather_than_short_circuit(self):
        # Seeing every failed check at once beats fixing them one per run.
        verdict = self.gate.evaluate(
            "bad", wf_result(card(sharpe=0.1, n=50, dd=-0.5, cost_drag=0.9,
                                  total_return=-0.1), n_folds=2, profitable=0),
            objective="crypto",
        )
        self.assertGreater(len(verdict.rejections), 2)


class GateHealthTests(unittest.TestCase):
    def test_rejection_profile_counts_reasons(self):
        gate = PromotionGate()
        gate.evaluate("a", wf_result(card(sharpe=0.1, total_return=-0.1)),
                      objective="crypto")
        profile = gate.rejection_profile()
        self.assertIn(Rejection.NEGATIVE_RETURN.value, profile)

    def test_high_promotion_rate_is_flagged_as_implausible(self):
        # A gate approving a third of a search's output is not filtering.
        gate = PromotionGate()
        for i in range(12):
            gate.evaluate(f"c{i}", wf_result(card(sharpe=6.0, n=2000, dd=-0.05)),
                          objective=f"objective_{i}")   # fresh objective each time
        health = gate.health()
        if health["promotion_rate"] > 0.3:
            self.assertTrue(health["warnings"])

    def test_zero_promotions_over_many_evaluations_is_flagged(self):
        gate = PromotionGate()
        for i in range(26):
            gate.evaluate(f"c{i}", wf_result(card(sharpe=0.05, total_return=-0.01)),
                          objective="crypto")
        self.assertTrue(any("nothing has been promoted" in w
                            for w in gate.health()["warnings"]))


if __name__ == "__main__":
    unittest.main()
