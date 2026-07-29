"""Configuration, risk profiles, and the live-trading arming gate.

Everything that can lose money is gated here rather than at the call site, so
that there is exactly one place to audit. Two independent conditions must both
hold before a real order can leave the process:

* ``SPINTRADER_LIVE_ENABLED=true`` -- the global arming switch, default off.
* the specific venue appears in ``SPINTRADER_LIVE_VENUES`` -- so arming Kraken
  never implicitly arms IBKR.

On top of that, :class:`RiskProfile` imposes notional caps that apply to live
orders regardless of what a strategy asks for.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields, replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from spintrader.core.types import TradingMode, VenueId, to_decimal


# --------------------------------------------------------------------------
# Env helpers
# --------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw.strip().lower() in _TRUTHY


def env_decimal(name: str, default: Decimal | str) -> Decimal:
    raw = os.getenv(name)
    return to_decimal(default) if raw is None or raw == "" else to_decimal(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


def env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return tuple(default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> int:
    """Load ``KEY=value`` pairs from a dotenv file into ``os.environ``.

    A deliberately small parser rather than a dependency: it handles comments,
    blank lines, ``export`` prefixes and quoted values, which covers everything
    this project's ``.env`` needs.

    Existing environment variables win by default, so a shell export can
    override the file without editing it. Returns the number of keys set.
    """
    p = Path(path)
    if not p.is_file():
        return 0

    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching surrounding quotes, but leave inner ones alone.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = value
        count += 1
    return count


def default_of(cls: type, name: str) -> Any:
    """Read a dataclass field's declared default.

    These config classes use ``slots=True``, which means ``cls.some_field`` is a
    slot *descriptor* rather than the default value -- so the obvious
    ``env_str("VAR", cls.some_field)`` silently yields
    ``"<member 'some_field' of ...>"``. Going through ``fields()`` keeps the
    declaration as the single source of truth without that trap.
    """
    for f in fields(cls):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:    # type: ignore[misc]
                return f.default_factory()          # type: ignore[misc]
            raise ValueError(f"{cls.__name__}.{name} has no default")
    raise KeyError(f"{cls.__name__} has no field {name!r}")


# --------------------------------------------------------------------------
# Risk profiles
# --------------------------------------------------------------------------

class Aggression(str, Enum):
    """User-facing trading style. Maps to concrete numeric risk limits."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Numeric risk limits derived from an :class:`Aggression` setting.

    The parameters are deliberately conservative relative to what a backtest
    would suggest is optimal. Full-Kelly sizing is theoretically growth-optimal
    but assumes the edge estimate is correct; on a live system whose edge is
    estimated by an LLM ensemble, a mis-estimated edge at full Kelly is ruin.
    ``kelly_fraction`` therefore tops out well below 1.0 even at maximum
    aggression.
    """
    aggression: Aggression

    # --- sizing ---
    kelly_fraction: Decimal            # multiplier on the Kelly-optimal size
    max_position_weight: Decimal       # single position as fraction of equity
    target_annual_vol: Decimal         # portfolio vol target for vol-scaling
    max_gross_exposure: Decimal        # sum |weights|; >1 implies leverage
    max_positions: int

    # --- loss limits ---
    stop_loss_pct: Decimal             # per-position hard stop
    daily_loss_limit: Decimal          # fraction of equity; halts trading for the day
    max_drawdown_limit: Decimal        # fraction from peak equity; trips the kill switch

    # --- behaviour ---
    min_confidence: Decimal            # decisions below this are dropped
    max_trades_per_day: int
    allow_shorts: bool
    allow_leverage: bool

    def scaled_for_regime(self, regime_risk: Decimal) -> "RiskProfile":
        """Return a profile scaled down for an unfavourable market regime.

        ``regime_risk`` runs 0 (benign) to 1 (crisis). Exposure is cut roughly
        linearly with it. This is the main channel by which the HMM regime
        layer influences sizing: the agents may stay bullish, but a crisis
        regime shrinks how much that bullishness is allowed to cost.
        """
        risk = max(Decimal("0"), min(Decimal("1"), to_decimal(regime_risk)))
        damp = Decimal("1") - (risk * Decimal("0.7"))
        return replace(
            self,
            kelly_fraction=self.kelly_fraction * damp,
            max_position_weight=self.max_position_weight * damp,
            max_gross_exposure=self.max_gross_exposure * damp,
        )


_PROFILES: dict[Aggression, RiskProfile] = {
    Aggression.CONSERVATIVE: RiskProfile(
        aggression=Aggression.CONSERVATIVE,
        kelly_fraction=Decimal("0.10"),
        max_position_weight=Decimal("0.10"),
        target_annual_vol=Decimal("0.08"),
        max_gross_exposure=Decimal("0.50"),
        max_positions=5,
        stop_loss_pct=Decimal("0.03"),
        daily_loss_limit=Decimal("0.01"),
        max_drawdown_limit=Decimal("0.05"),
        min_confidence=Decimal("0.70"),
        max_trades_per_day=3,
        allow_shorts=False,
        allow_leverage=False,
    ),
    Aggression.MODERATE: RiskProfile(
        aggression=Aggression.MODERATE,
        kelly_fraction=Decimal("0.20"),
        max_position_weight=Decimal("0.15"),
        target_annual_vol=Decimal("0.12"),
        max_gross_exposure=Decimal("0.80"),
        max_positions=8,
        stop_loss_pct=Decimal("0.05"),
        daily_loss_limit=Decimal("0.02"),
        max_drawdown_limit=Decimal("0.10"),
        min_confidence=Decimal("0.62"),
        max_trades_per_day=6,
        allow_shorts=False,
        allow_leverage=False,
    ),
    Aggression.BALANCED: RiskProfile(
        aggression=Aggression.BALANCED,
        kelly_fraction=Decimal("0.30"),
        max_position_weight=Decimal("0.20"),
        target_annual_vol=Decimal("0.18"),
        max_gross_exposure=Decimal("1.00"),
        max_positions=10,
        stop_loss_pct=Decimal("0.07"),
        daily_loss_limit=Decimal("0.03"),
        max_drawdown_limit=Decimal("0.15"),
        min_confidence=Decimal("0.55"),
        max_trades_per_day=10,
        allow_shorts=True,
        allow_leverage=False,
    ),
    Aggression.GROWTH: RiskProfile(
        aggression=Aggression.GROWTH,
        kelly_fraction=Decimal("0.40"),
        max_position_weight=Decimal("0.28"),
        target_annual_vol=Decimal("0.25"),
        max_gross_exposure=Decimal("1.30"),
        max_positions=12,
        stop_loss_pct=Decimal("0.10"),
        daily_loss_limit=Decimal("0.05"),
        max_drawdown_limit=Decimal("0.22"),
        min_confidence=Decimal("0.50"),
        max_trades_per_day=15,
        allow_shorts=True,
        allow_leverage=True,
    ),
    Aggression.AGGRESSIVE: RiskProfile(
        aggression=Aggression.AGGRESSIVE,
        kelly_fraction=Decimal("0.50"),
        max_position_weight=Decimal("0.35"),
        target_annual_vol=Decimal("0.35"),
        max_gross_exposure=Decimal("1.75"),
        max_positions=15,
        stop_loss_pct=Decimal("0.15"),
        daily_loss_limit=Decimal("0.08"),
        max_drawdown_limit=Decimal("0.30"),
        min_confidence=Decimal("0.45"),
        max_trades_per_day=25,
        allow_shorts=True,
        allow_leverage=True,
    ),
}


def risk_profile(aggression: Aggression | str) -> RiskProfile:
    """Look up the numeric limits for an aggression setting."""
    key = Aggression(aggression) if not isinstance(aggression, Aggression) else aggression
    return _PROFILES[key]


# --------------------------------------------------------------------------
# LLM tiers
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Local-model routing for the GB10.

    Two tiers because a single decision cycle makes dozens of LLM calls. Most
    are routine (summarise this filing, extract these numbers) and belong on a
    fast model; only the synthesis and adjudication steps justify loading the
    99 GB GLM-4.5-Air, which on a 121 GB unified-memory box crowds out
    everything else while resident.
    """
    base_url: str = "http://10.0.0.62:11434"
    quick_model: str = "qwen3:30b-a3b"
    deep_model: str = "MichelRosselli/GLM-4.5-Air:Q6_K"
    embed_model: str = "nomic-embed-text"
    request_timeout_s: int = 300
    max_retries: int = 3
    temperature: float = 0.3
    # Guardrail: a runaway debate loop can otherwise burn hours of GPU time.
    max_calls_per_cycle: int = 120
    max_tokens_per_cycle: int = 1_500_000

    @classmethod
    def from_env(cls) -> "LLMConfig":
        d = lambda n: default_of(cls, n)  # noqa: E731
        return cls(
            base_url=env_str("SPINTRADER_OLLAMA_URL", d("base_url")),
            quick_model=env_str("SPINTRADER_QUICK_MODEL", d("quick_model")),
            deep_model=env_str("SPINTRADER_DEEP_MODEL", d("deep_model")),
            embed_model=env_str("SPINTRADER_EMBED_MODEL", d("embed_model")),
            request_timeout_s=env_int("SPINTRADER_LLM_TIMEOUT", d("request_timeout_s")),
            max_retries=env_int("SPINTRADER_LLM_RETRIES", d("max_retries")),
            temperature=float(env_str("SPINTRADER_LLM_TEMPERATURE", str(d("temperature")))),
            max_calls_per_cycle=env_int("SPINTRADER_MAX_LLM_CALLS", d("max_calls_per_cycle")),
            max_tokens_per_cycle=env_int("SPINTRADER_MAX_LLM_TOKENS", d("max_tokens_per_cycle")),
        )


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class SecretFileError(RuntimeError):
    """A configured secret file is missing or unreadable."""


def read_secret_file(path: str | Path) -> str:
    """Read a single-line secret from disk, stripping the trailing newline.

    Preferred over putting the value in ``.env``: the secret stays in one file
    with its own permissions, never appears in a shell environment (where any
    child process and ``/proc/<pid>/environ`` can see it), and never risks
    being pasted into a chat or a commit.
    """
    p = Path(path)
    try:
        value = p.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SecretFileError(f"cannot read secret file {p}: {exc}") from exc
    if not value:
        raise SecretFileError(f"secret file {p} is empty")
    return value


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """TimescaleDB on the GB10 plus a parquet lake for research datasets."""
    host: str = "10.0.0.62"
    port: int = 5433
    database: str = "spintrader"
    user: str = "spintrader"
    password: str = ""
    data_root: Path = Path("/home/spinner/data/spintrader")
    pool_min: int = 1
    pool_max: int = 10

    @classmethod
    def from_env(cls) -> "StorageConfig":
        d = lambda n: default_of(cls, n)  # noqa: E731

        # A password file wins over an inline password. The file lives on the
        # GB10 beside the database it unlocks and is written by the
        # provisioning script, so the credential never has to travel.
        password_file = env_str("SPINTRADER_DB_PASSWORD_FILE", "")
        password = (
            read_secret_file(password_file) if password_file
            else env_str("SPINTRADER_DB_PASSWORD", "")
        )

        return cls(
            host=env_str("SPINTRADER_DB_HOST", d("host")),
            port=env_int("SPINTRADER_DB_PORT", d("port")),
            database=env_str("SPINTRADER_DB_NAME", d("database")),
            user=env_str("SPINTRADER_DB_USER", d("user")),
            password=password,
            data_root=Path(env_str("SPINTRADER_DATA_ROOT", str(d("data_root")))),
        )

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def redacted_dsn(self) -> str:
        """Safe to log."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"


# --------------------------------------------------------------------------
# Live-trading gate
# --------------------------------------------------------------------------

class LiveTradingDisarmed(RuntimeError):
    """Raised when live routing is attempted while the gate is closed."""


@dataclass(frozen=True, slots=True)
class LiveGate:
    """The arming switch for real order flow.

    Deliberately awkward to satisfy: two env vars plus a per-venue notional cap.
    The caps below are absolute dollar ceilings applied *after* all strategy
    sizing, so a sizing bug cannot produce an order larger than these.
    """
    enabled: bool = False
    venues: frozenset[VenueId] = field(default_factory=frozenset)
    max_order_notional: dict[VenueId, Decimal] = field(default_factory=dict)
    max_daily_notional: dict[VenueId, Decimal] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LiveGate":
        raw_venues = env_list("SPINTRADER_LIVE_VENUES")
        venues: set[VenueId] = set()
        for name in raw_venues:
            try:
                venues.add(VenueId(name.lower()))
            except ValueError:
                raise ValueError(
                    f"SPINTRADER_LIVE_VENUES contains unknown venue {name!r}; "
                    f"valid values: {[v.value for v in VenueId]}"
                ) from None
        return cls(
            enabled=env_bool("SPINTRADER_LIVE_ENABLED", False),
            venues=frozenset(venues),
            # Micro-live defaults sized for a ~$600 book: a single bad order
            # can cost at most $25 on Kraken, $10 on the IBKR cash account.
            max_order_notional={
                VenueId.KRAKEN: env_decimal("SPINTRADER_KRAKEN_MAX_ORDER", "25"),
                VenueId.IBKR: env_decimal("SPINTRADER_IBKR_MAX_ORDER", "10"),
            },
            max_daily_notional={
                VenueId.KRAKEN: env_decimal("SPINTRADER_KRAKEN_MAX_DAILY", "150"),
                VenueId.IBKR: env_decimal("SPINTRADER_IBKR_MAX_DAILY", "50"),
            },
        )

    def is_armed(self, venue: VenueId) -> bool:
        return self.enabled and venue in self.venues

    def check(self, venue: VenueId, mode: TradingMode, notional: Decimal | None = None) -> None:
        """Raise unless this order is permitted to go live. No-op for simulated modes."""
        if mode.is_simulated:
            return
        if not self.enabled:
            raise LiveTradingDisarmed(
                "live trading is disarmed (set SPINTRADER_LIVE_ENABLED=true to arm)"
            )
        if venue not in self.venues:
            armed = sorted(v.value for v in self.venues) or ["<none>"]
            raise LiveTradingDisarmed(
                f"venue {venue.value!r} is not armed for live trading; armed: {armed} "
                f"(add it to SPINTRADER_LIVE_VENUES)"
            )
        if notional is not None:
            cap = self.max_order_notional.get(venue)
            if cap is not None and notional > cap:
                raise LiveTradingDisarmed(
                    f"order notional {notional} exceeds the live cap {cap} for {venue.value}"
                )


# --------------------------------------------------------------------------
# Top-level settings
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    """Whole-system configuration, assembled from the environment."""
    mode: TradingMode = TradingMode.PAPER
    aggression: Aggression = Aggression.MODERATE
    base_currency: str = "USD"
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    live: LiveGate = field(default_factory=LiveGate)
    # Universe the system is allowed to trade at all. An empty tuple means
    # "no restriction", which is only sensible in backtests.
    crypto_universe: tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD")
    equity_universe: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")
    log_level: str = "INFO"
    dry_run: bool = False

    # --- paper/live fidelity -------------------------------------------
    # IBKR seeds paper accounts with 1,000,000 CAD on 3.33x margin, against a
    # real account holding ~100 USD in a *cash* account. Sizing against the
    # reported paper equity would validate strategies that cannot be run:
    # leverage that does not exist, positions that are unaffordable, and round
    # trips faster than T+1 settlement allows.
    #
    # When set, the risk engine sizes against this figure instead of the
    # venue-reported equity, in every simulated mode. None means trust the
    # venue -- correct for live, wrong for paper.
    paper_equity_override: Decimal | None = None
    # Enforce cash-account rules (no leverage, no shorting, settled funds only)
    # even where the venue would permit more.
    enforce_cash_account: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        raw_override = env_str("SPINTRADER_PAPER_EQUITY", "")
        return cls(
            mode=TradingMode(env_str("SPINTRADER_MODE", TradingMode.PAPER.value).lower()),
            aggression=Aggression(env_str("SPINTRADER_AGGRESSION", Aggression.MODERATE.value).lower()),
            base_currency=env_str("SPINTRADER_BASE_CURRENCY", "USD"),
            llm=LLMConfig.from_env(),
            storage=StorageConfig.from_env(),
            live=LiveGate.from_env(),
            crypto_universe=env_list("SPINTRADER_CRYPTO_UNIVERSE", default_of(cls, "crypto_universe")),
            equity_universe=env_list("SPINTRADER_EQUITY_UNIVERSE", default_of(cls, "equity_universe")),
            log_level=env_str("SPINTRADER_LOG_LEVEL", "INFO"),
            dry_run=env_bool("SPINTRADER_DRY_RUN", False),
            paper_equity_override=to_decimal(raw_override) if raw_override else None,
            enforce_cash_account=env_bool("SPINTRADER_ENFORCE_CASH_ACCOUNT", True),
        )

    @property
    def risk(self) -> RiskProfile:
        return risk_profile(self.aggression)

    def effective_equity(self, venue_equity: Decimal) -> Decimal:
        """Equity the risk engine should size against.

        In live mode the venue is authoritative. In paper and backtest, an
        override (when configured) wins, so that simulated results describe a
        book the real account could actually hold.
        """
        if self.mode is TradingMode.LIVE or self.paper_equity_override is None:
            return venue_equity
        return self.paper_equity_override

    def describe(self) -> dict[str, Any]:
        """Human-readable summary for startup logs. Contains no secrets."""
        return {
            "mode": self.mode.value,
            "aggression": self.aggression.value,
            "live_enabled": self.live.enabled,
            "live_venues": sorted(v.value for v in self.live.venues),
            "quick_model": self.llm.quick_model,
            "deep_model": self.llm.deep_model,
            "database": self.storage.redacted_dsn(),
            "max_position_weight": str(self.risk.max_position_weight),
            "max_drawdown_limit": str(self.risk.max_drawdown_limit),
            "dry_run": self.dry_run,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton. ``refresh=True`` re-reads the environment."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings.from_env()
    return _settings


__all__ = [
    "Aggression", "LLMConfig", "LiveGate", "LiveTradingDisarmed", "RiskProfile",
    "SecretFileError", "Settings", "StorageConfig", "get_settings", "risk_profile",
    "env_bool", "env_decimal", "env_int", "env_list", "env_str",
    "default_of", "load_env_file", "read_secret_file",
]
