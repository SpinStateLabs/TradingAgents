"""Kraken venue client.

Kraken's API carries a decade of accumulated naming history, and getting it
wrong produces silent misvaluation rather than an error:

* Assets listed before ~2018 carry a class prefix -- ``XXBT``, ``XETH``,
  ``ZUSD``, ``ZCAD`` -- where ``X`` marks a cryptocurrency and ``Z`` a fiat.
  Assets listed since do not: ``BNB``, ``USDC``, ``SOL``. Any code that assumes
  one convention mishandles half the book.
* ``XBT`` is Bitcoin. A symbol map keyed on ``BTC`` silently misses it.
* Pair names follow the same split: ``XXBTZUSD`` but ``BNBUSD``. Building pair
  names by string concatenation therefore fails unpredictably, so this client
  resolves them from ``AssetPairs`` instead of guessing.

Everything venue-specific stops here. The strategy layer sees ``BTC-USD``.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Mapping

import requests

from spintrader.core.config import Settings, env_str
from spintrader.core.types import (
    AssetClass, Balance, Fill, Instrument, Order, OrderStatus, OrderType,
    Position, Quote, Side, TimeInForce, VenueId, to_decimal, utcnow,
)
from spintrader.venues.base import (
    AccountSnapshot, OrderRejected, Venue, VenueError,
)
from spintrader.venues.kraken_auth import KrakenAuthError, KrakenCredentials

log = logging.getLogger(__name__)

API_BASE = "https://api.kraken.com"

# Kraken's legacy class prefixes -> canonical ticker. Only assets listed before
# the naming change carry these; newer listings appear verbatim.
_LEGACY_ASSETS = {
    "XXBT": "BTC", "XBT": "BTC",
    "XETH": "ETH", "XLTC": "LTC", "XXRP": "XRP", "XXLM": "XLM",
    "XXMR": "XMR", "XZEC": "ZEC", "XREP": "REP", "XETC": "ETC",
    "XMLN": "MLN", "XXDG": "DOGE", "XDG": "DOGE",
    "ZUSD": "USD", "ZEUR": "EUR", "ZCAD": "CAD", "ZGBP": "GBP",
    "ZJPY": "JPY", "ZAUD": "AUD", "ZCHF": "CHF",
}

FIAT = frozenset({"USD", "EUR", "CAD", "GBP", "JPY", "AUD", "CHF"})


def normalise_asset(code: str) -> str:
    """Map a Kraken asset code to its canonical ticker.

    ``XXBT`` -> ``BTC``, ``ZCAD`` -> ``CAD``, ``BNB`` -> ``BNB``. Handles the
    ``.S``/``.M``/``.F`` staking and earn suffixes Kraken appends to bonded
    balances, which would otherwise read as separate assets.
    """
    code = code.strip().upper()
    for suffix in (".S", ".M", ".F", ".B", ".HOLD"):
        if code.endswith(suffix):
            code = code[: -len(suffix)]
            break
    return _LEGACY_ASSETS.get(code, code)


class KrakenVenue(Venue):
    """Spot trading and market data on Kraken."""

    venue_id = VenueId.KRAKEN

    def __init__(
        self,
        settings: Settings | None = None,
        credentials: KrakenCredentials | None = None,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._credentials = credentials
        self._session = session or requests.Session()
        self._pairs: dict[str, dict[str, Any]] = {}     # canonical symbol -> pair info
        self._instruments: dict[str, Instrument] = {}

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        if self._credentials is None:
            key, secret = env_str("KRAKEN_API_KEY", ""), env_str("KRAKEN_API_SECRET", "")
            if key and secret:
                self._credentials = KrakenCredentials(key, secret)
            else:
                # Public data still works; anything account-related will raise.
                log.warning("kraken: no credentials, running in public-data-only mode")
        self._load_pairs()

    def _disconnect(self) -> None:
        self._session.close()

    @property
    def authenticated(self) -> bool:
        return self._credentials is not None

    def _require_auth(self) -> KrakenCredentials:
        if self._credentials is None:
            raise VenueError(
                "kraken: this operation needs credentials; set KRAKEN_API_KEY "
                "and KRAKEN_API_SECRET"
            )
        return self._credentials

    # -- transport ---------------------------------------------------------

    def _public(self, endpoint: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = self._session.get(
            f"{API_BASE}/0/public/{endpoint}", params=dict(params or {}), timeout=30
        )
        response.raise_for_status()
        return self._unwrap(response.json(), endpoint)

    def _private(self, endpoint: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        creds = self._require_auth()
        path = f"/0/private/{endpoint}"
        headers, body = creds.signed_request(path, params)
        response = self._session.post(f"{API_BASE}{path}", headers=headers, data=body, timeout=30)
        response.raise_for_status()
        return self._unwrap(response.json(), endpoint)

    @staticmethod
    def _unwrap(payload: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
        errors = payload.get("error") or []
        if errors:
            # Kraken returns HTTP 200 with an error array, so a naive client
            # treats failures as successes.
            if any("Permission denied" in e for e in errors):
                raise VenueError(
                    f"kraken {endpoint}: permission denied -- the API key is "
                    f"missing a required permission"
                )
            if any(e.startswith("EOrder:") or e.startswith("EFunding:") for e in errors):
                raise OrderRejected(f"kraken {endpoint}: {', '.join(errors)}")
            raise VenueError(f"kraken {endpoint}: {', '.join(errors)}")
        return payload.get("result", {})

    # -- instruments -------------------------------------------------------

    def _load_pairs(self) -> None:
        """Cache the tradable pair definitions.

        Pair names are resolved from the API rather than constructed, because
        the legacy prefixes make ``XXBTZUSD`` and ``BNBUSD`` both valid forms
        with no rule connecting them to their canonical symbols.
        """
        result = self._public("AssetPairs")
        for pair_name, info in result.items():
            if pair_name.endswith(".d"):       # dark-pool duplicates
                continue
            base = normalise_asset(info.get("base", ""))
            quote = normalise_asset(info.get("quote", ""))
            symbol = f"{base}-{quote}"
            self._pairs[symbol] = {**info, "_pair_name": pair_name,
                                   "_base": base, "_quote": quote}
        log.info("kraken: loaded %d tradable pairs", len(self._pairs))

    def resolve(self, symbol: str) -> Instrument:
        symbol = symbol.upper()
        if not self._pairs:
            self._load_pairs()
        info = self._pairs.get(symbol)
        if info is None:
            raise VenueError(
                f"kraken does not list {symbol!r}"
                + (f" (did you mean one of {self._suggest(symbol)}?)" if self._suggest(symbol) else "")
            )

        if symbol in self._instruments:
            return self._instruments[symbol]

        pair_decimals = int(info.get("pair_decimals", 2))
        lot_decimals = int(info.get("lot_decimals", 8))
        fees = info.get("fees") or [[0, 0.26]]
        fees_maker = info.get("fees_maker") or [[0, 0.16]]

        instrument = Instrument(
            symbol=symbol,
            asset_class=AssetClass.FX if info["_base"] in FIAT else AssetClass.CRYPTO,
            venue=self.venue_id,
            venue_symbol=info["_pair_name"],
            base_currency=info["_base"],
            quote_currency=info["_quote"],
            price_increment=Decimal(1).scaleb(-pair_decimals),
            qty_increment=Decimal(1).scaleb(-lot_decimals),
            min_qty=to_decimal(info.get("ordermin", "0")),
            min_notional=to_decimal(info.get("costmin", "0")),
            maker_fee=to_decimal(fees_maker[0][1]) / Decimal(100),
            taker_fee=to_decimal(fees[0][1]) / Decimal(100),
        )
        self._instruments[symbol] = instrument
        return instrument

    def _suggest(self, symbol: str) -> list[str]:
        base = symbol.split("-", 1)[0]
        return sorted(s for s in self._pairs if s.startswith(f"{base}-"))[:5]

    # -- market data -------------------------------------------------------

    def get_quote(self, instrument: Instrument) -> Quote:
        result = self._public("Ticker", {"pair": instrument.venue_symbol})
        if not result:
            raise VenueError(f"kraken: no ticker for {instrument.symbol}")
        data = next(iter(result.values()))
        return Quote(
            instrument_key=instrument.key,
            ts=utcnow(),
            bid=to_decimal(data["b"][0]),
            ask=to_decimal(data["a"][0]),
            bid_size=to_decimal(data["b"][2]),
            ask_size=to_decimal(data["a"][2]),
        )

    # -- account -----------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        raw = self._private("Balance")
        balances: dict[str, Balance] = {}
        positions: dict[str, Position] = {}

        # Staked and bonded balances arrive under suffixed codes; fold them
        # into the base asset so the book shows one line per asset.
        merged: dict[str, Decimal] = {}
        for code, amount in raw.items():
            value = to_decimal(amount)
            if value == 0:
                continue
            merged[normalise_asset(code)] = merged.get(normalise_asset(code), Decimal(0)) + value

        base_ccy = self.settings.base_currency
        equity = Decimal("0")

        for asset, amount in merged.items():
            if asset in FIAT:
                balances[asset] = Balance(
                    currency=asset, total=amount, available=amount,
                    venue=self.venue_id, ts=utcnow(),
                )
                equity += amount if asset == base_ccy else self._to_base(amount, asset, base_ccy)
            else:
                # A spot crypto holding is a position, not cash.
                symbol = f"{asset}-{base_ccy}"
                key = f"{self.venue_id.value}:{symbol}"
                mark = self._safe_mark(symbol)
                positions[key] = Position(
                    instrument_key=key, qty=amount,
                    avg_cost=Decimal("0"),      # Kraken does not report cost basis
                    last_price=mark,
                )
                if mark is not None:
                    equity += amount * mark

        return AccountSnapshot(
            venue=self.venue_id, balances=balances, positions=positions,
            equity=equity, ts=utcnow(),
        )

    def _safe_mark(self, symbol: str) -> Decimal | None:
        try:
            return self.get_quote(self.resolve(symbol)).mid
        except (VenueError, requests.RequestException):
            return None

    def _to_base(self, amount: Decimal, currency: str, base: str) -> Decimal:
        """Convert a fiat balance into the base currency for equity reporting."""
        if currency == base:
            return amount
        # Kraken quotes USD/CAD, not CAD/USD -- divide rather than multiply.
        for symbol, invert in ((f"{base}-{currency}", True), (f"{currency}-{base}", False)):
            try:
                rate = self.get_quote(self.resolve(symbol)).mid
            except (VenueError, requests.RequestException):
                continue
            if rate and rate > 0:
                return amount / rate if invert else amount * rate
        log.warning("kraken: cannot convert %s to %s; excluded from equity", currency, base)
        return Decimal("0")

    # -- orders ------------------------------------------------------------

    # Kraken accepts GTC, IOC and GTD. It has no FOK, and no session-scoped
    # DAY for spot -- both are mapped to IOC, the nearest honest equivalent,
    # rather than silently downgrading them to a resting GTC order.
    _TIF = {
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "IOC",
        TimeInForce.DAY: "GTC",
    }

    _ORDER_TYPE = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop-loss",
        OrderType.STOP_LIMIT: "stop-loss-limit",
    }

    def _transmit(self, order: Order) -> Order:
        params: dict[str, Any] = {
            "pair": order.instrument.venue_symbol,
            "type": order.side.value,
            "ordertype": self._ORDER_TYPE[order.order_type],
            "volume": str(order.qty),
            "userref": order.client_order_id[-9:].lstrip("-") or None,
        }
        if order.limit_price is not None:
            params["price"] = str(order.limit_price)
        if order.stop_price is not None:
            key = "price2" if order.order_type is OrderType.STOP_LIMIT else "price"
            params[key] = str(order.stop_price)
        tif = self._TIF.get(order.time_in_force)
        if tif and tif != "GTC":
            params["timeinforce"] = tif
        params = {k: v for k, v in params.items() if v is not None}

        try:
            result = self._private("AddOrder", params)
        except OrderRejected as exc:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(exc)
            raise

        txids = result.get("txid") or []
        order.venue_order_id = txids[0] if txids else None
        order.status = OrderStatus.OPEN
        order.updated_at = utcnow()
        log.info("kraken: submitted %s -> %s", self.describe_order(order), order.venue_order_id)
        return order

    def cancel(self, order: Order) -> Order:
        if order.venue_order_id is None:
            raise VenueError(f"order {order.order_id} has no venue id; was it transmitted?")
        self._private("CancelOrder", {"txid": order.venue_order_id})
        order.status = OrderStatus.CANCELED
        order.updated_at = utcnow()
        return order

    def validate_order(self, order: Order) -> dict[str, Any]:
        """Ask Kraken to parse an order without placing it.

        Useful for verifying permissions and order construction. ``validate``
        makes this a no-op at the exchange -- nothing is queued or filled.
        """
        params = {
            "pair": order.instrument.venue_symbol,
            "type": order.side.value,
            "ordertype": self._ORDER_TYPE[order.order_type],
            "volume": str(order.qty),
            "validate": "true",
        }
        if order.limit_price is not None:
            params["price"] = str(order.limit_price)
        return self._private("AddOrder", params)


__all__ = ["KrakenVenue", "normalise_asset", "FIAT"]
