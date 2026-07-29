"""Tests for configuration and, above all, the live-trading gate.

The gate tests are the most important in the repo. Everything else here can
fail and cost time; the gate failing open costs money. Each test below encodes
one way the gate must refuse to open.
"""

from __future__ import annotations

import os
import unittest
from decimal import Decimal
from unittest import mock

from spintrader.core.config import (
    Aggression, LLMConfig, LiveGate, LiveTradingDisarmed, Settings,
    StorageConfig, get_settings, risk_profile,
)
from spintrader.core.types import TradingMode, VenueId

D = Decimal


def with_env(**kwargs: str):
    """Replace the environment entirely, so ambient vars cannot leak in."""
    return mock.patch.dict(os.environ, kwargs, clear=True)


class LiveGateTests(unittest.TestCase):
    def test_default_is_disarmed(self):
        with with_env():
            gate = LiveGate.from_env()
        self.assertFalse(gate.enabled)
        self.assertEqual(gate.venues, frozenset())
        self.assertFalse(gate.is_armed(VenueId.KRAKEN))

    def test_simulated_modes_bypass_the_gate(self):
        gate = LiveGate(enabled=False)
        for mode in (TradingMode.PAPER, TradingMode.BACKTEST):
            with self.subTest(mode=mode):
                gate.check(VenueId.KRAKEN, mode)   # must not raise

    def test_live_blocked_when_globally_disabled(self):
        gate = LiveGate(enabled=False, venues=frozenset({VenueId.KRAKEN}))
        with self.assertRaises(LiveTradingDisarmed) as ctx:
            gate.check(VenueId.KRAKEN, TradingMode.LIVE)
        self.assertIn("disarmed", str(ctx.exception))

    def test_enabling_one_venue_does_not_arm_another(self):
        # The failure mode this guards against: arming Kraken for crypto and
        # discovering the equity strategy started sending real IBKR orders.
        gate = LiveGate(enabled=True, venues=frozenset({VenueId.KRAKEN}))
        gate.check(VenueId.KRAKEN, TradingMode.LIVE)
        with self.assertRaises(LiveTradingDisarmed) as ctx:
            gate.check(VenueId.IBKR, TradingMode.LIVE)
        self.assertIn("not armed", str(ctx.exception))

    def test_notional_cap_enforced(self):
        gate = LiveGate(
            enabled=True,
            venues=frozenset({VenueId.KRAKEN}),
            max_order_notional={VenueId.KRAKEN: D("25")},
        )
        gate.check(VenueId.KRAKEN, TradingMode.LIVE, notional=D("24.99"))
        gate.check(VenueId.KRAKEN, TradingMode.LIVE, notional=D("25"))
        with self.assertRaises(LiveTradingDisarmed) as ctx:
            gate.check(VenueId.KRAKEN, TradingMode.LIVE, notional=D("25.01"))
        self.assertIn("exceeds the live cap", str(ctx.exception))

    def test_env_arming_roundtrip(self):
        with with_env(
            SPINTRADER_LIVE_ENABLED="true",
            SPINTRADER_LIVE_VENUES="kraken, ibkr",
            SPINTRADER_KRAKEN_MAX_ORDER="30",
        ):
            gate = LiveGate.from_env()
        self.assertTrue(gate.is_armed(VenueId.KRAKEN))
        self.assertTrue(gate.is_armed(VenueId.IBKR))
        self.assertEqual(gate.max_order_notional[VenueId.KRAKEN], D("30"))

    def test_typo_in_venue_list_fails_loudly(self):
        # Silently ignoring an unknown venue would make "krakn" read as
        # "nothing armed" -- or worse, hide that a venue the user meant to arm
        # never was.
        with with_env(SPINTRADER_LIVE_ENABLED="true", SPINTRADER_LIVE_VENUES="krakn"):
            with self.assertRaises(ValueError) as ctx:
                LiveGate.from_env()
        self.assertIn("unknown venue", str(ctx.exception))

    def test_only_explicit_truthy_values_arm(self):
        for raw, expected in [
            ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
            ("false", False), ("0", False), ("no", False), ("", False),
            ("maybe", False), ("tru", False),
        ]:
            with self.subTest(raw=raw):
                with with_env(SPINTRADER_LIVE_ENABLED=raw):
                    self.assertEqual(LiveGate.from_env().enabled, expected)


class RiskProfileTests(unittest.TestCase):
    def test_all_aggressions_resolve(self):
        for a in Aggression:
            with self.subTest(aggression=a):
                self.assertEqual(risk_profile(a).aggression, a)

    def test_accepts_string_alias(self):
        self.assertEqual(risk_profile("aggressive").aggression, Aggression.AGGRESSIVE)

    def test_limits_are_monotonic_in_aggression(self):
        order = [
            Aggression.CONSERVATIVE, Aggression.MODERATE, Aggression.BALANCED,
            Aggression.GROWTH, Aggression.AGGRESSIVE,
        ]
        profiles = [risk_profile(a) for a in order]
        for attr in ("kelly_fraction", "max_position_weight", "target_annual_vol",
                     "max_gross_exposure", "stop_loss_pct", "daily_loss_limit",
                     "max_drawdown_limit"):
            values = [getattr(p, attr) for p in profiles]
            with self.subTest(attr=attr):
                self.assertEqual(values, sorted(values), f"{attr} must not decrease with aggression")
        # Confidence threshold moves the other way: bolder profiles act on weaker signals.
        confidences = [p.min_confidence for p in profiles]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_kelly_never_reaches_full(self):
        # Full Kelly on an LLM-estimated edge is a ruin risk, not an
        # optimisation. Nothing in the profile table may cross 0.5.
        for a in Aggression:
            with self.subTest(aggression=a):
                self.assertLessEqual(risk_profile(a).kelly_fraction, D("0.5"))

    def test_conservative_forbids_shorts_and_leverage(self):
        p = risk_profile(Aggression.CONSERVATIVE)
        self.assertFalse(p.allow_shorts)
        self.assertFalse(p.allow_leverage)

    def test_regime_scaling_reduces_exposure(self):
        base = risk_profile(Aggression.BALANCED)
        crisis = base.scaled_for_regime(D("1"))
        self.assertLess(crisis.kelly_fraction, base.kelly_fraction)
        self.assertLess(crisis.max_position_weight, base.max_position_weight)
        self.assertLess(crisis.max_gross_exposure, base.max_gross_exposure)
        # Loss limits are not relaxed by regime scaling.
        self.assertEqual(crisis.max_drawdown_limit, base.max_drawdown_limit)

    def test_benign_regime_is_a_noop(self):
        base = risk_profile(Aggression.BALANCED)
        self.assertEqual(base.scaled_for_regime(D("0")), base)

    def test_regime_risk_is_clamped(self):
        base = risk_profile(Aggression.BALANCED)
        self.assertEqual(base.scaled_for_regime(D("-5")), base.scaled_for_regime(D("0")))
        self.assertEqual(base.scaled_for_regime(D("99")), base.scaled_for_regime(D("1")))

    def test_scaling_never_inverts_exposure(self):
        for a in Aggression:
            with self.subTest(aggression=a):
                scaled = risk_profile(a).scaled_for_regime(D("1"))
                self.assertGreater(scaled.max_position_weight, 0)


class StorageConfigTests(unittest.TestCase):
    def test_dsn_built_from_parts(self):
        cfg = StorageConfig(host="h", port=5433, database="db", user="u", password="p")
        self.assertEqual(cfg.dsn, "postgresql://u:p@h:5433/db")

    def test_redacted_dsn_hides_password(self):
        cfg = StorageConfig(host="h", port=5433, database="db", user="u", password="hunter2")
        self.assertNotIn("hunter2", cfg.redacted_dsn())
        self.assertIn("***", cfg.redacted_dsn())


class SecretFileTests(unittest.TestCase):
    """File-based secrets keep the DB password out of .env and out of the
    process environment, where any child process can read it."""

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "pgpass")

    def write(self, content: str) -> str:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return self.path

    def test_reads_and_strips_trailing_newline(self):
        from spintrader.core.config import read_secret_file
        self.write("s3cret\n")
        self.assertEqual(read_secret_file(self.path), "s3cret")

    def test_missing_file_raises(self):
        from spintrader.core.config import SecretFileError, read_secret_file
        with self.assertRaises(SecretFileError):
            read_secret_file(os.path.join(self._dir.name, "nope"))

    def test_empty_file_raises(self):
        from spintrader.core.config import SecretFileError, read_secret_file
        self.write("   \n")
        with self.assertRaises(SecretFileError):
            read_secret_file(self.path)

    def test_storage_config_prefers_the_file(self):
        self.write("from-file")
        with with_env(
            SPINTRADER_DB_PASSWORD_FILE=self.path,
            SPINTRADER_DB_PASSWORD="from-env",
        ):
            self.assertEqual(StorageConfig.from_env().password, "from-file")

    def test_storage_config_falls_back_to_inline(self):
        with with_env(SPINTRADER_DB_PASSWORD="from-env"):
            self.assertEqual(StorageConfig.from_env().password, "from-env")

    def test_blank_file_var_is_ignored(self):
        # An empty SPINTRADER_DB_PASSWORD_FILE must not be treated as a path.
        with with_env(SPINTRADER_DB_PASSWORD_FILE="", SPINTRADER_DB_PASSWORD="inline"):
            self.assertEqual(StorageConfig.from_env().password, "inline")

    def test_bad_path_surfaces_at_startup(self):
        from spintrader.core.config import SecretFileError
        with with_env(SPINTRADER_DB_PASSWORD_FILE="/nonexistent/pgpass"):
            with self.assertRaises(SecretFileError):
                StorageConfig.from_env()

    def test_file_sourced_password_stays_out_of_describe(self):
        self.write("hunter2")
        with with_env(SPINTRADER_DB_PASSWORD_FILE=self.path):
            self.assertNotIn("hunter2", repr(Settings.from_env().describe()))


class LoadEnvFileTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, ".env")

    def write(self, content: str) -> str:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return self.path

    def test_parses_pairs_and_skips_comments(self):
        from spintrader.core.config import load_env_file
        self.write("# a comment\n\nFOO=bar\nBAZ=qux\n")
        with with_env():
            self.assertEqual(load_env_file(self.path), 2)
            self.assertEqual(os.environ["FOO"], "bar")

    def test_strips_quotes_and_export_prefix(self):
        from spintrader.core.config import load_env_file
        self.write("export A='one'\nB=\"two\"\n")
        with with_env():
            load_env_file(self.path)
            self.assertEqual((os.environ["A"], os.environ["B"]), ("one", "two"))

    def test_existing_environment_wins_by_default(self):
        from spintrader.core.config import load_env_file
        self.write("FOO=from_file\n")
        with with_env(FOO="from_shell"):
            load_env_file(self.path)
            self.assertEqual(os.environ["FOO"], "from_shell")

    def test_override_flag(self):
        from spintrader.core.config import load_env_file
        self.write("FOO=from_file\n")
        with with_env(FOO="from_shell"):
            load_env_file(self.path, override=True)
            self.assertEqual(os.environ["FOO"], "from_file")

    def test_value_containing_equals_is_preserved(self):
        # Base64 secrets routinely end in '=' padding.
        from spintrader.core.config import load_env_file
        self.write("SECRET=abc==\n")
        with with_env():
            load_env_file(self.path)
            self.assertEqual(os.environ["SECRET"], "abc==")

    def test_missing_file_is_not_an_error(self):
        from spintrader.core.config import load_env_file
        self.assertEqual(load_env_file(os.path.join(self._dir.name, "absent")), 0)


class SettingsTests(unittest.TestCase):
    def test_defaults_are_safe(self):
        with with_env():
            s = Settings.from_env()
        self.assertEqual(s.mode, TradingMode.PAPER)
        self.assertFalse(s.live.enabled)
        self.assertEqual(s.aggression, Aggression.MODERATE)

    def test_env_overrides(self):
        with with_env(SPINTRADER_MODE="backtest", SPINTRADER_AGGRESSION="aggressive"):
            s = Settings.from_env()
        self.assertEqual(s.mode, TradingMode.BACKTEST)
        self.assertEqual(s.risk.aggression, Aggression.AGGRESSIVE)

    def test_universe_parsed_from_csv(self):
        with with_env(SPINTRADER_CRYPTO_UNIVERSE="BTC-USD, ETH-USD ,SOL-USD"):
            s = Settings.from_env()
        self.assertEqual(s.crypto_universe, ("BTC-USD", "ETH-USD", "SOL-USD"))

    def test_bad_mode_rejected(self):
        with with_env(SPINTRADER_MODE="yolo"):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_describe_leaks_no_secrets(self):
        with with_env(SPINTRADER_DB_PASSWORD="hunter2"):
            described = repr(Settings.from_env().describe())
        self.assertNotIn("hunter2", described)

    def test_singleton_refresh(self):
        with with_env(SPINTRADER_AGGRESSION="conservative"):
            first = get_settings(refresh=True)
            self.assertEqual(first.aggression, Aggression.CONSERVATIVE)
        with with_env(SPINTRADER_AGGRESSION="growth"):
            self.assertIs(get_settings(), first)               # cached
            self.assertEqual(get_settings(refresh=True).aggression, Aggression.GROWTH)


class EffectiveEquityTests(unittest.TestCase):
    """Paper/live fidelity.

    IBKR hands the paper account 1,000,000 CAD on 3.33x margin against a real
    account holding ~100 USD of cash. Sizing against the reported figure would
    validate strategies that cannot be run at all.
    """

    def test_live_always_trusts_the_venue(self):
        with with_env(SPINTRADER_MODE="live", SPINTRADER_PAPER_EQUITY="100"):
            s = Settings.from_env()
        self.assertEqual(s.effective_equity(D("1000000")), D("1000000"))

    def test_paper_uses_the_override(self):
        with with_env(SPINTRADER_MODE="paper", SPINTRADER_PAPER_EQUITY="100"):
            s = Settings.from_env()
        self.assertEqual(s.effective_equity(D("1000000")), D("100"))

    def test_backtest_uses_the_override(self):
        with with_env(SPINTRADER_MODE="backtest", SPINTRADER_PAPER_EQUITY="250.50"):
            s = Settings.from_env()
        self.assertEqual(s.effective_equity(D("1000000")), D("250.50"))

    def test_paper_without_override_trusts_the_venue(self):
        with with_env(SPINTRADER_MODE="paper"):
            s = Settings.from_env()
        self.assertIsNone(s.paper_equity_override)
        self.assertEqual(s.effective_equity(D("1000000")), D("1000000"))

    def test_blank_override_is_not_zero(self):
        # An empty env var must mean "unset", never "size against nothing".
        with with_env(SPINTRADER_MODE="paper", SPINTRADER_PAPER_EQUITY=""):
            s = Settings.from_env()
        self.assertIsNone(s.paper_equity_override)

    def test_cash_account_enforced_by_default(self):
        with with_env():
            self.assertTrue(Settings.from_env().enforce_cash_account)

    def test_cash_account_can_be_disabled(self):
        with with_env(SPINTRADER_ENFORCE_CASH_ACCOUNT="false"):
            self.assertFalse(Settings.from_env().enforce_cash_account)


class LLMConfigTests(unittest.TestCase):
    def test_defaults_point_at_the_gb10(self):
        with with_env():
            cfg = LLMConfig.from_env()
        self.assertIn("10.0.0.62", cfg.base_url)
        self.assertNotEqual(cfg.quick_model, cfg.deep_model)

    def test_cycle_budgets_are_positive(self):
        with with_env():
            cfg = LLMConfig.from_env()
        self.assertGreater(cfg.max_calls_per_cycle, 0)
        self.assertGreater(cfg.max_tokens_per_cycle, 0)


if __name__ == "__main__":
    unittest.main()
