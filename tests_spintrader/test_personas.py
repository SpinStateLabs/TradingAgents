"""Tests for the persona roster and panel.

The behaviours worth defending here are the ones that keep the ensemble honest:

* **Abstention works and is not a neutral vote.** A roster of mostly
  inapplicable personas must not dilute the informed ones toward inaction.
* **Method diversity is real.** Tests assert the roster actually spans lenses
  and horizons, because a roster that drifts toward all-momentum would still
  pass every functional test while being worthless as an ensemble.
* **Single-lens agreement is discounted.** Five personas agreeing because they
  all read the same evidence is one opinion.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from spintrader.agents.panel import (
    PanelVerdict, PersonaPanel, PersonaVote, abstain,
)
from spintrader.agents.personas.roster import (
    BERKSHIRE, BURRY, MANDELBROT, POLICY_HEADLINE, SIMONS, TALEB, THIEL,
    default_roster,
)
from spintrader.agents.personas.spec import (
    Horizon, Lens, PersonaRegistry, PersonaSpec,
)
from spintrader.core.types import Action, AssetClass

D = Decimal
CRYPTO = AssetClass.CRYPTO
EQUITY = AssetClass.EQUITY


def vote(key, action=Action.BUY, confidence="0.8", lenses=(), rationale="because"):
    return PersonaVote(persona_key=key, action=action, confidence=D(confidence),
                       rationale=rationale, lenses=tuple(lenses))


class RosterCompositionTests(unittest.TestCase):
    """The roster's value is diversity of method, not fame."""

    def setUp(self):
        self.roster = default_roster()

    def test_roster_is_populated(self):
        self.assertGreaterEqual(len(self.roster), 12)

    def test_keys_are_unique(self):
        self.assertEqual(len(self.roster.keys()), len(set(self.roster.keys())))

    def test_duplicate_registration_rejected(self):
        with self.assertRaises(ValueError):
            self.roster.register(BURRY)

    def test_roster_spans_many_lenses(self):
        # A roster that drifted to all-statistical would pass every other test
        # while being useless as an ensemble.
        lenses = {lens for spec in self.roster.all() for lens in spec.lenses}
        self.assertGreaterEqual(len(lenses), 6, f"only {len(lenses)} lenses covered")

    def test_roster_spans_many_horizons(self):
        horizons = {spec.native_horizon for spec in self.roster.all()}
        self.assertGreaterEqual(len(horizons), 3)

    def test_contrarian_axis_is_spread(self):
        values = [float(s.contrarian) for s in self.roster.all()]
        self.assertLess(min(values), 0.3)      # trend followers exist
        self.assertGreater(max(values), 0.85)  # deep contrarians exist

    def test_every_persona_states_how_it_could_be_wrong(self):
        # A methodology with no falsification condition rationalises any outcome.
        for spec in self.roster.all():
            with self.subTest(persona=spec.key):
                self.assertTrue(spec.invalidation,
                                f"{spec.key} has no invalidation conditions")

    def test_every_persona_admits_its_failure_modes(self):
        for spec in self.roster.all():
            with self.subTest(persona=spec.key):
                self.assertTrue(spec.known_failure_modes,
                                f"{spec.key} claims no weaknesses")

    def test_every_persona_names_its_attribution(self):
        for spec in self.roster.all():
            with self.subTest(persona=spec.key):
                self.assertTrue(spec.attribution)

    def test_unknown_persona_lookup_lists_options(self):
        with self.assertRaises(KeyError) as ctx:
            self.roster.get("nope")
        self.assertIn("registered", str(ctx.exception))


class ApplicabilityTests(unittest.TestCase):
    def test_equity_only_persona_abstains_on_crypto(self):
        ok, reason = BURRY.applies_to(CRYPTO, Horizon.MONTHS)
        self.assertFalse(ok)
        self.assertIn("crypto", reason)

    def test_long_horizon_persona_abstains_on_intraday(self):
        # Buffett-style analysis has nothing to say about an hourly decision.
        ok, reason = BERKSHIRE.applies_to(EQUITY, Horizon.INTRADAY)
        self.assertFalse(ok)
        self.assertIn("horizon", reason)

    def test_statistical_persona_handles_short_horizons(self):
        ok, _ = SIMONS.applies_to(CRYPTO, Horizon.MINUTES)
        self.assertTrue(ok)

    def test_abstention_reasons_are_stated(self):
        roster = default_roster()
        abstentions = roster.abstentions(CRYPTO, Horizon.INTRADAY)
        self.assertTrue(abstentions)
        for key, reason in abstentions.items():
            with self.subTest(persona=key):
                self.assertTrue(reason.strip(), f"{key} abstained silently")

    def test_hourly_crypto_still_has_a_usable_panel(self):
        # The realistic case for this book. If everyone abstains the design is
        # broken.
        roster = default_roster()
        applicable = roster.applicable(CRYPTO, Horizon.INTRADAY)
        self.assertGreaterEqual(len(applicable), 3,
                                f"only {len(applicable)} personas can trade hourly crypto")

    def test_hourly_crypto_panel_is_lens_diverse(self):
        roster = default_roster()
        coverage = roster.lens_coverage(CRYPTO, Horizon.INTRADAY)
        self.assertGreaterEqual(len(coverage), 3, f"lens coverage too narrow: {coverage}")

    def test_horizon_fit_peaks_at_native(self):
        self.assertEqual(SIMONS.horizon_fit(SIMONS.native_horizon), D("1"))

    def test_horizon_fit_decays_with_distance(self):
        near = BERKSHIRE.horizon_fit(Horizon.MONTHS)
        far = BERKSHIRE.horizon_fit(Horizon.DAYS)
        self.assertGreater(near, far)

    def test_horizon_fit_never_negative(self):
        self.assertGreaterEqual(BERKSHIRE.horizon_fit(Horizon.MINUTES), D("0"))


class ReliabilityTests(unittest.TestCase):
    def test_reliability_defaults_to_one(self):
        self.assertEqual(BURRY.reliability, D("1"))

    def test_reliability_is_clamped(self):
        self.assertEqual(BURRY.with_reliability(D("5")).reliability, D("1"))
        self.assertEqual(BURRY.with_reliability(D("-1")).reliability, D("0"))

    def test_registry_updates_reliability(self):
        roster = default_roster()
        roster.update_reliability("burry", D("0.25"))
        self.assertEqual(roster.get("burry").reliability, D("0.25"))

    def test_unreliable_personas_can_be_filtered_out(self):
        roster = default_roster()
        roster.update_reliability("simons", D("0.1"))
        keys = [s.key for s in roster.applicable(CRYPTO, Horizon.INTRADAY,
                                                 min_reliability=D("0.5"))]
        self.assertNotIn("simons", keys)

    def test_applicable_is_ordered_by_fit_times_reliability(self):
        roster = default_roster()
        ranked = roster.applicable(CRYPTO, Horizon.DAYS)
        scores = [s.horizon_fit(Horizon.DAYS) * s.reliability for s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class SystemPromptTests(unittest.TestCase):
    def test_prompt_includes_method_and_invalidation(self):
        prompt = BURRY.system_prompt()
        self.assertIn("Michael Burry", prompt)
        self.assertIn("WRONG if", prompt)
        self.assertIn("ABSTAIN", prompt)

    def test_prompt_instructs_against_borrowing_other_methods(self):
        # The ensemble's value depends on members staying different.
        self.assertIn("different", TALEB.system_prompt())

    def test_prompt_surfaces_failure_modes(self):
        self.assertIn("weaknesses", MANDELBROT.system_prompt())

    def test_prompt_is_derived_not_stored(self):
        # Behaviour follows the spec, so a spec change cannot leave a stale
        # prompt behind.
        modified = BURRY.with_reliability(D("0.5"))
        self.assertEqual(modified.system_prompt(), BURRY.system_prompt())


class VoteTests(unittest.TestCase):
    def test_buy_is_positive_direction(self):
        self.assertEqual(vote("simons", Action.BUY, "0.8").direction, D("0.8"))

    def test_sell_is_negative_direction(self):
        self.assertEqual(vote("simons", Action.SELL, "0.8").direction, D("-0.8"))

    def test_hold_is_abstention(self):
        self.assertTrue(vote("simons", Action.HOLD, "0.9").abstained)

    def test_zero_confidence_is_abstention(self):
        self.assertTrue(vote("simons", Action.BUY, "0").abstained)

    def test_abstain_helper_carries_a_reason(self):
        v = abstain("berkshire", "no cash flows to value")
        self.assertTrue(v.abstained)
        self.assertIn("cash flows", v.rationale)


class PanelAggregationTests(unittest.TestCase):
    def setUp(self):
        self.roster = default_roster()
        self.panel = PersonaPanel(self.roster)

    def test_unanimous_buy(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL]),
             vote("taleb", Action.BUY, "0.7", [Lens.TAIL]),
             vote("thorp", Action.BUY, "0.75", [Lens.STATISTICAL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertEqual(verdict.action, Action.BUY)
        self.assertGreater(verdict.confidence, D("0.4"))
        self.assertLess(verdict.dispersion, D("0.2"))

    def test_opposed_votes_of_equal_weight_cancel(self):
        # simons and mandelbrot are both native to DAYS, so their weights match
        # and equal-and-opposite votes genuinely cancel.
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL]),
             vote("mandelbrot", Action.SELL, "0.8", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertEqual(verdict.net_direction, D("0"))
        self.assertGreater(verdict.dispersion, D("0.3"))

    def test_off_native_horizon_vote_carries_less_weight(self):
        """Opposed votes do NOT cancel symmetrically across horizons.

        Taleb is native to MONTHS, so at a DAYS horizon his vote counts half.
        A months-horizon tail-risk process asked about a days-horizon decision
        is being stretched, and should not fully offset a persona operating in
        its own timeframe.
        """
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL]),
             vote("taleb", Action.SELL, "0.8", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertGreater(verdict.net_direction, D("0"))   # simons prevails
        weights = {v.persona_key: v.weight for v in verdict.votes}
        self.assertGreater(weights["simons"], weights["taleb"])

    def test_the_same_pair_cancels_at_taleb_native_horizon(self):
        # At MONTHS, Taleb is native and Simons is stretched, so the asymmetry
        # reverses -- confirming it is horizon fit and not a persona bias.
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL]),
             vote("taleb", Action.SELL, "0.8", [Lens.TAIL])],
            CRYPTO, Horizon.MONTHS,
        )
        self.assertLess(verdict.net_direction, D("0"))      # taleb prevails

    def test_disagreement_triggers_escalation(self):
        # The deep model should adjudicate contested calls; agreement does not
        # need it.
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("taleb", Action.SELL, "0.9", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertTrue(verdict.escalate)

    def test_abstentions_do_not_dilute_the_denominator(self):
        """The key aggregation choice.

        Counting abstentions as neutral would let inapplicable personas drag an
        informed panel toward inaction.
        """
        strong = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("thorp", Action.BUY, "0.9", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        with_abstainers = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("thorp", Action.BUY, "0.9", [Lens.TAIL]),
             abstain("berkshire", "no cash flows"),
             abstain("burry", "no filings")],
            CRYPTO, Horizon.DAYS,
        )
        self.assertEqual(strong.net_direction, with_abstainers.net_direction)

    def test_no_participants_means_hold(self):
        verdict = self.panel.aggregate(
            [abstain("berkshire", "nothing to value"),
             abstain("burry", "no filings")],
            CRYPTO, Horizon.INTRADAY,
        )
        self.assertEqual(verdict.action, Action.HOLD)
        self.assertEqual(verdict.confidence, D("0"))
        self.assertEqual(verdict.participating, 0)

    def test_single_lens_agreement_is_discounted(self):
        # Three personas agreeing because they all read statistics is one
        # opinion wearing three hats.
        same_lens = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("thorp", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("turtle", Action.BUY, "0.9", [Lens.STATISTICAL])],
            CRYPTO, Horizon.DAYS,
        )
        diverse = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("taleb", Action.BUY, "0.9", [Lens.TAIL]),
             vote("dalio", Action.BUY, "0.9", [Lens.MACRO])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertLess(same_lens.confidence, diverse.confidence)

    def test_single_lens_agreement_escalates(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("turtle", Action.BUY, "0.9", [Lens.STATISTICAL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertTrue(verdict.escalate)

    def test_thin_panel_is_discounted(self):
        thin = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertLess(thin.confidence, D("0.9"))

    def test_unreliable_persona_counts_for_less(self):
        full = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("taleb", Action.SELL, "0.9", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        self.roster.update_reliability("taleb", D("0.1"))
        downweighted = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("taleb", Action.SELL, "0.9", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        self.assertGreater(downweighted.net_direction, full.net_direction)

    def test_unknown_persona_vote_is_discarded(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL]),
             vote("not_a_persona", Action.BUY, "0.9")],
            CRYPTO, Horizon.DAYS,
        )
        self.assertEqual(verdict.participating, 1)

    def test_caller_cannot_inflate_its_own_weight(self):
        # Weights come from the registry, not from the vote.
        rigged = vote("simons", Action.BUY, "0.9", [Lens.STATISTICAL])
        rigged.weight = D("1000")
        verdict = self.panel.aggregate([rigged], CRYPTO, Horizon.DAYS)
        self.assertLessEqual(verdict.votes[0].weight, D("1"))

    def test_confidence_stays_in_range(self):
        verdict = self.panel.aggregate(
            [vote(k, Action.BUY, "1.0", [Lens.STATISTICAL, Lens.TAIL, Lens.MACRO])
             for k in ("simons", "thorp", "taleb", "dalio", "mandelbrot")],
            CRYPTO, Horizon.DAYS,
        )
        self.assertGreaterEqual(verdict.confidence, D("0"))
        self.assertLessEqual(verdict.confidence, D("1"))


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.panel = PersonaPanel(default_roster())

    def test_contributions_record_every_vote(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL], "pattern held"),
             abstain("berkshire", "no cash flows")],
            CRYPTO, Horizon.DAYS,
        )
        contributions = verdict.contributions()
        self.assertEqual(len(contributions["votes"]), 2)
        self.assertIn("berkshire", contributions["abstentions"])

    def test_decision_carries_attribution(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL]),
             vote("taleb", Action.BUY, "0.7", [Lens.TAIL])],
            CRYPTO, Horizon.DAYS,
        )
        decision = self.panel.to_decision(verdict, "kraken:BTC-USD",
                                          Horizon.DAYS, regime="calm")
        self.assertEqual(decision.action, Action.BUY)
        self.assertEqual(decision.regime, "calm")
        self.assertIn("votes", decision.contributions)
        self.assertIn("simons", decision.rationale)

    def test_decision_confidence_is_a_valid_probability(self):
        verdict = self.panel.aggregate(
            [vote("simons", Action.BUY, "0.8", [Lens.STATISTICAL])],
            CRYPTO, Horizon.DAYS,
        )
        decision = self.panel.to_decision(verdict, "kraken:BTC-USD", Horizon.DAYS)
        self.assertGreaterEqual(decision.confidence, D("0"))
        self.assertLessEqual(decision.confidence, D("1"))

    def test_empty_panel_produces_an_honest_rationale(self):
        verdict = self.panel.aggregate([], CRYPTO, Horizon.DAYS)
        decision = self.panel.to_decision(verdict, "kraken:BTC-USD", Horizon.DAYS)
        self.assertEqual(decision.action, Action.HOLD)
        self.assertIn("no persona voted", decision.rationale)


class PolicyHeadlineTests(unittest.TestCase):
    """The persona with no documented practitioner gets extra scrutiny."""

    def test_attribution_is_honest_about_having_no_single_source(self):
        self.assertIn("no single documented practitioner",
                      POLICY_HEADLINE.attribution)

    def test_admits_it_may_be_noise_on_this_book(self):
        text = " ".join(POLICY_HEADLINE.known_failure_modes)
        self.assertIn("noise", text)

    def test_has_a_hard_time_limit_in_its_method(self):
        # Without one it becomes an unfalsifiable directional bet.
        text = " ".join(POLICY_HEADLINE.method) + " ".join(POLICY_HEADLINE.invalidation)
        self.assertIn("time limit", text)

    def test_short_horizon_and_impatient(self):
        self.assertEqual(POLICY_HEADLINE.native_horizon, Horizon.INTRADAY)
        self.assertLess(POLICY_HEADLINE.patience, D("0.3"))


if __name__ == "__main__":
    unittest.main()
