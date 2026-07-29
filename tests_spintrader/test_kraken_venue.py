"""Tests for the Kraken venue client. All HTTP is mocked.

Asset-name normalisation gets the most attention. Kraken's legacy class
prefixes (XXBT, ZCAD) coexist with modern bare codes (BNB, USDC), and code
that assumes either convention alone misvalues part of the book *silently* --
a balance keyed 'ZCAD' simply fails to match a 'CAD' lookup and reads as zero.
That is the failure this file exists to prevent.
"""

from __future__ import annotations

import unittest
from decimal import Decimal
from unittest import mock

import requests

from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    AssetClass, Order, OrderStatus, OrderType, Side, TradingMode, VenueId,
)
from spintrader.venues.base import OrderRejected, VenueError
from spintrader.venues.kraken import KrakenVenue, normalise_asset
from spintrader.venues.kraken_auth import KrakenCredentials

D = Decimal
import base64
DUMMY_SECRET = base64.b64encode(b"\x02" * 64).decode()

ASSET_PAIRS = {
    "XXBTZUSD": {
        "base": "XXBT", "quote": "ZUSD", "pair_decimals": 1, "lot_decimals": 8,
        "ordermin": "0.00005", "costmin": "5",
        "fees": [[0, 0.26]], "fees_maker": [[0, 0.16]],
    },
    "BNBUSD": {
        "base": "BNB", "quote": "ZUSD", "pair_decimals": 3, "lot_decimals": 8,
        "ordermin": "0.02", "costmin": "5",
        "fees": [[0, 0.26]], "fees_maker": [[0, 0.16]],
    },
    "USDCAD": {
        "base": "ZUSD", "quote": "ZCAD", "pair_decimals": 5, "lot_decimals": 8,
        "ordermin": "5", "costmin": "5",
        "fees": [[0, 0.20]], "fees_maker": [[0, 0.20]],
    },
    "XXBTZUSD.d": {  # dark pool duplicate, must be ignored
        "base": "XXBT", "quote": "ZUSD", "pair_decimals": 1, "lot_decimals": 8,
    },
}


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.PAPER, aggression=Aggression.BALANCED,
                live=LiveGate(enabled=False), base_currency="USD")
    base.update(kw)
    return Settings(**base)


class FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body, self.status_code = body, status
        self.text = str(body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def make_venue(get_bodies=None, post_bodies=None, **kw) -> tuple[KrakenVenue, mock.Mock]:
    session = mock.Mock(spec=requests.Session)
    get_queue = list(get_bodies or [{"error": [], "result": ASSET_PAIRS}])
    post_queue = list(post_bodies or [])

    def do_get(url, **_kw):
        body = get_queue.pop(0) if len(get_queue) > 1 else get_queue[0]
        return FakeResponse(body)

    def do_post(url, **_kw):
        return FakeResponse(post_queue.pop(0) if post_queue else {"error": [], "result": {}})

    session.get.side_effect = do_get
    session.post.side_effect = do_post

    venue = KrakenVenue(
        settings=kw.pop("settings", settings()),
        credentials=KrakenCredentials("key", DUMMY_SECRET),
        session=session,
    )
    venue.connect()
    return venue, session


class NormaliseAssetTests(unittest.TestCase):
    def test_legacy_crypto_prefix_stripped(self):
        self.assertEqual(normalise_asset("XXBT"), "BTC")
        self.assertEqual(normalise_asset("XETH"), "ETH")

    def test_legacy_fiat_prefix_stripped(self):
        self.assertEqual(normalise_asset("ZUSD"), "USD")
        self.assertEqual(normalise_asset("ZCAD"), "CAD")

    def test_xbt_is_bitcoin(self):
        # A map keyed on 'BTC' alone silently misses Kraken's XBT.
        self.assertEqual(normalise_asset("XBT"), "BTC")

    def test_modern_codes_pass_through(self):
        for code in ("BNB", "USDC", "SOL"):
            with self.subTest(code=code):
                self.assertEqual(normalise_asset(code), code)

    def test_staking_suffixes_folded_into_the_base_asset(self):
        # Otherwise a staked balance reads as a separate, unpriceable asset.
        self.assertEqual(normalise_asset("ETH.S"), "ETH")
        self.assertEqual(normalise_asset("USDC.M"), "USDC")
        self.assertEqual(normalise_asset("XXBT.F"), "BTC")

    def test_case_and_whitespace_tolerated(self):
        self.assertEqual(normalise_asset("  xxbt "), "BTC")


class ResolveTests(unittest.TestCase):
    def test_legacy_pair_gets_a_canonical_symbol(self):
        venue, _ = make_venue()
        inst = venue.resolve("BTC-USD")
        self.assertEqual(inst.venue_symbol, "XXBTZUSD")   # venue's name
        self.assertEqual(inst.symbol, "BTC-USD")          # ours

    def test_modern_pair_resolves(self):
        venue, _ = make_venue()
        self.assertEqual(venue.resolve("BNB-USD").venue_symbol, "BNBUSD")

    def test_increments_derived_from_decimals(self):
        venue, _ = make_venue()
        inst = venue.resolve("BTC-USD")
        self.assertEqual(inst.price_increment, D("0.1"))    # pair_decimals 1
        self.assertEqual(inst.qty_increment, D("1E-8"))     # lot_decimals 8

    def test_minimums_and_fees_carried_through(self):
        venue, _ = make_venue()
        inst = venue.resolve("BTC-USD")
        self.assertEqual(inst.min_qty, D("0.00005"))
        self.assertEqual(inst.min_notional, D("5"))
        self.assertEqual(inst.taker_fee, D("0.0026"))
        self.assertEqual(inst.maker_fee, D("0.0016"))

    def test_fiat_pair_classified_as_fx(self):
        venue, _ = make_venue()
        self.assertEqual(venue.resolve("USD-CAD").asset_class, AssetClass.FX)

    def test_crypto_pair_classified_as_crypto(self):
        venue, _ = make_venue()
        self.assertEqual(venue.resolve("BTC-USD").asset_class, AssetClass.CRYPTO)

    def test_dark_pool_duplicates_ignored(self):
        venue, _ = make_venue()
        self.assertNotIn(".d", venue.resolve("BTC-USD").venue_symbol)

    def test_unlisted_symbol_raises_with_suggestions(self):
        # BNB-CAD genuinely does not exist on Kraken; the error should say so
        # and point at what does.
        venue, _ = make_venue()
        with self.assertRaises(VenueError) as ctx:
            venue.resolve("BNB-CAD")
        message = str(ctx.exception)
        self.assertIn("does not list", message)
        self.assertIn("BNB-USD", message)

    def test_resolution_is_cached(self):
        venue, _ = make_venue()
        self.assertIs(venue.resolve("BTC-USD"), venue.resolve("BTC-USD"))


class ErrorHandlingTests(unittest.TestCase):
    """Kraken returns HTTP 200 with an error array; naive clients read failures
    as successes."""

    def test_permission_error_is_explicit(self):
        venue, _ = make_venue(post_bodies=[{"error": ["EGeneral:Permission denied"]}])
        with self.assertRaises(VenueError) as ctx:
            venue.snapshot()
        self.assertIn("permission denied", str(ctx.exception).lower())

    def test_order_errors_map_to_order_rejected(self):
        venue, _ = make_venue(post_bodies=[{"error": ["EOrder:Insufficient funds"]}])
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.01"),
                      order_type=OrderType.LIMIT, limit_price=D("50000"))
        with self.assertRaises(OrderRejected):
            venue._transmit(order)

    def test_generic_errors_raise_venue_error(self):
        venue, _ = make_venue(post_bodies=[{"error": ["EService:Unavailable"]}])
        with self.assertRaises(VenueError):
            venue.snapshot()

    def test_empty_error_array_is_success(self):
        venue, _ = make_venue(post_bodies=[{"error": [], "result": {"ZUSD": "100"}}])
        self.assertEqual(venue.snapshot().cash("USD"), D("100"))


class SnapshotTests(unittest.TestCase):
    def _venue_with_balances(self, balances: dict, ticker: dict | None = None):
        ticker = ticker or {"error": [], "result": {
            "XXBTZUSD": {"b": ["60000.0", "1", "1.0"], "a": ["60001.0", "1", "1.0"]},
        }}
        return make_venue(
            get_bodies=[{"error": [], "result": ASSET_PAIRS}, ticker],
            post_bodies=[{"error": [], "result": balances}],
        )

    def test_fiat_becomes_a_balance(self):
        venue, _ = self._venue_with_balances({"ZUSD": "708.90"})
        snap = venue.snapshot()
        self.assertEqual(snap.cash("USD"), D("708.90"))
        self.assertEqual(snap.positions, {})

    def test_legacy_fiat_code_is_normalised(self):
        # The exact bug that made a 1000 CAD balance read as unpriceable.
        venue, _ = self._venue_with_balances({"ZCAD": "1000"})
        self.assertIn("CAD", venue.snapshot().balances)

    def test_crypto_becomes_a_position_not_cash(self):
        venue, _ = self._venue_with_balances({"BNB": "0.50949893"})
        snap = venue.snapshot()
        self.assertEqual(snap.balances, {})
        self.assertIn("kraken:BNB-USD", snap.positions)
        self.assertEqual(snap.positions["kraken:BNB-USD"].qty, D("0.50949893"))

    def test_zero_balances_omitted(self):
        venue, _ = self._venue_with_balances({"ZUSD": "0", "BNB": "0"})
        snap = venue.snapshot()
        self.assertEqual(snap.balances, {})
        self.assertEqual(snap.positions, {})

    def test_staked_and_spot_balances_are_merged(self):
        # Otherwise the same asset appears twice and equity double-counts or
        # misses one line.
        venue, _ = self._venue_with_balances({"ETH": "1.0", "ETH.S": "2.0"})
        positions = venue.snapshot().positions
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions["kraken:ETH-USD"].qty, D("3.0"))

    def test_unpriceable_asset_does_not_break_the_snapshot(self):
        venue, _ = make_venue(
            get_bodies=[{"error": [], "result": ASSET_PAIRS},
                        {"error": ["EQuery:Unknown asset pair"], "result": {}}],
            post_bodies=[{"error": [], "result": {"WEIRDCOIN": "5"}}],
        )
        snap = venue.snapshot()   # must not raise
        self.assertIsNone(snap.positions["kraken:WEIRDCOIN-USD"].last_price)


class OrderTransmissionTests(unittest.TestCase):
    def test_limit_order_parameters(self):
        venue, session = make_venue(post_bodies=[{"error": [], "result": {"txid": ["OABC-123"]}}])
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.001"),
                      order_type=OrderType.LIMIT, limit_price=D("60000"))
        venue._transmit(order)
        body = session.post.call_args.kwargs["data"]
        self.assertIn("pair=XXBTZUSD", body)
        self.assertIn("type=buy", body)
        self.assertIn("ordertype=limit", body)
        self.assertIn("price=60000", body)
        self.assertEqual(order.venue_order_id, "OABC-123")
        self.assertEqual(order.status, OrderStatus.OPEN)

    def test_market_order_omits_price(self):
        venue, session = make_venue(post_bodies=[{"error": [], "result": {"txid": ["OX"]}}])
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.001"))
        venue._transmit(order)
        self.assertNotIn("price=", session.post.call_args.kwargs["data"])

    def test_rejection_marks_the_order(self):
        venue, _ = make_venue(post_bodies=[{"error": ["EOrder:Cost minimum not met"]}])
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.0001"))
        with self.assertRaises(OrderRejected):
            venue._transmit(order)
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertIn("Cost minimum", order.reject_reason)

    def test_validate_order_sets_the_validate_flag(self):
        # Proves permissions without placing anything.
        venue, session = make_venue(post_bodies=[{"error": [], "result": {"descr": {}}}])
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.001"),
                      order_type=OrderType.LIMIT, limit_price=D("20000"))
        venue.validate_order(order)
        self.assertIn("validate=true", session.post.call_args.kwargs["data"])

    def test_cancel_requires_a_venue_id(self):
        venue, _ = make_venue()
        order = Order(venue.resolve("BTC-USD"), Side.BUY, D("0.001"))
        with self.assertRaises(VenueError):
            venue.cancel(order)


class AuthRequirementTests(unittest.TestCase):
    def test_public_data_works_without_credentials(self):
        session = mock.Mock(spec=requests.Session)
        session.get.return_value = FakeResponse({"error": [], "result": ASSET_PAIRS})
        venue = KrakenVenue(settings=settings(), session=session)
        with mock.patch.dict("os.environ", {}, clear=True):
            venue.connect()
        self.assertFalse(venue.authenticated)
        self.assertEqual(venue.resolve("BTC-USD").symbol, "BTC-USD")

    def test_account_access_without_credentials_raises_clearly(self):
        session = mock.Mock(spec=requests.Session)
        session.get.return_value = FakeResponse({"error": [], "result": ASSET_PAIRS})
        venue = KrakenVenue(settings=settings(), session=session)
        with mock.patch.dict("os.environ", {}, clear=True):
            venue.connect()
            with self.assertRaises(VenueError) as ctx:
                venue.snapshot()
        self.assertIn("KRAKEN_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
