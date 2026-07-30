"""News / social sentiment ingestion, aggregated to bar close-times (task 17).

Why this exists
---------------
The price feeds (:mod:`spintrader.data.kraken_feed`, ``kraken_ws``,
``kraken_trades``) tell the system *what* the market did. They cannot tell it
*what the crowd was saying while it happened*. This feed adds that second
channel: a per-interval sentiment series, built from public social sources, that
lines up one-to-one with the OHLCV bars a strategy already sees.

Shape, mirrored from :mod:`spintrader.data.kraken_trades`
--------------------------------------------------------
This module is deliberately the sentiment analogue of the trades feed, so the
two read the same way:

* a raw tick type -- :class:`SentimentMention` here, ``TradeTick`` there;
* a fetch function with an **injectable fetcher** -- :func:`fetch_sentiment`,
  so a test needs no network and a caller can swap the source;
* **HTTP-200-with-error handling** in the default fetcher (StockTwits, like
  Kraken, answers 200 with an error body that a naive client reads as empty
  success);
* **UTC timestamps and Decimal at the boundary** -- scores and weights become
  ``Decimal`` as they arrive, times are timezone-aware UTC;
* an aggregator that buckets raw mentions into a per-interval series, holding
  the forming bucket until a later mention proves it complete -- exactly the
  rule ``MinuteBarAggregator`` applies to trades.

Correctness rules (identical in spirit to the trades feed)
----------------------------------------------------------
* **A score's ``ts`` is the bucket CLOSE time.** A mention at 10:03:45 belongs
  to the ``[10:03:00, 10:04:00)`` bucket, whose close -- the first instant it is
  fully observable -- is 10:04:00. This is what lets a strategy look sentiment up
  by the current bar's close-time without reading the future.
* **The forming bucket is never emitted** until a mention in a later bucket
  proves it done. Aligning sentiment to a bar you can already see is the whole
  point; emitting a half-formed bucket would leak the future.
* **Sentiment is optional.** Every network path degrades to an empty result
  rather than raising, so a strategy that wants sentiment as a *secondary* signal
  is never taken down by a flaky social endpoint.

Sentiment scoring
-----------------
Two signals are combined per mention. When a source labels a message itself --
StockTwits' ``Bullish``/``Bearish`` tag -- that label is authoritative. When it
does not -- a Reddit post -- polarity is estimated from a small keyword lexicon
over the title and body. The parsing of each source is *adapted* from the
existing :mod:`tradingagents.dataflows.reddit` and
:mod:`tradingagents.dataflows.stocktwits` fetchers (same fields, same graceful
degradation), but produces structured mentions instead of prompt text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from spintrader.core.types import ensure_utc, to_decimal

log = logging.getLogger(__name__)

# Public, keyless endpoints -- the same ones the tradingagents fetchers use.
STOCKTWITS_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
REDDIT_API = "https://www.reddit.com/r/{sub}/search.json"
USER_AGENT = "spintrader/0.1 (sentiment feed)"

# Finance subreddits, ordered roughly by signal density (see reddit.py).
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing", "CryptoCurrency")

ONE = Decimal("1")
ZERO = Decimal("0")
NEG_ONE = Decimal("-1")

# A deliberately small, auditable polarity lexicon. It is not a sentiment model;
# it is a transparent fallback for sources that do not label their own messages.
# Kept short on purpose: a long opaque list would be a model masquerading as a
# lookup, and the point here is that the estimate is inspectable.
_BULLISH = frozenset({
    "buy", "long", "bull", "bullish", "moon", "calls", "pump", "breakout",
    "rally", "up", "surge", "gain", "gains", "green", "beat", "strong", "hodl",
})
_BEARISH = frozenset({
    "sell", "short", "bear", "bearish", "puts", "dump", "crash", "drop",
    "down", "fall", "loss", "losses", "red", "weak", "miss", "rug", "fear",
})


def _clamp_score(value: Decimal) -> Decimal:
    """Bound a polarity to [-1, 1]."""
    if value > ONE:
        return ONE
    if value < NEG_ONE:
        return NEG_ONE
    return value


def score_text(text: str) -> Decimal:
    """Estimate polarity in [-1, 1] from a small keyword lexicon.

    ``(bull - bear) / (bull + bear)`` when any lexicon word is present, else a
    neutral zero. Deterministic and offline, so it is safe in a test and cheap
    at minute cadence.
    """
    if not text:
        return ZERO
    bull = bear = 0
    for raw in text.lower().split():
        word = raw.strip(".,!?:;\"'()[]#$@").lstrip("$")
        if word in _BULLISH:
            bull += 1
        elif word in _BEARISH:
            bear += 1
    total = bull + bear
    if total == 0:
        return ZERO
    return _clamp_score(Decimal(bull - bear) / Decimal(total))


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SentimentMention:
    """A single labelled mention from a social source.

    ``score`` is this one message's polarity in [-1, 1]; ``weight`` is its
    engagement (upvotes, comments, likes) and is never negative -- it measures
    *how loud* the mention was, which the aggregator uses to volume-weight the
    interval score.
    """
    ts: datetime
    symbol: str
    score: Decimal            # polarity of this mention, in [-1, 1]
    weight: Decimal           # engagement/volume weight, >= 0
    source: str               # "stocktwits" | "reddit:wallstreetbets" | ...

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        object.__setattr__(self, "score", _clamp_score(to_decimal(self.score)))
        w = to_decimal(self.weight)
        object.__setattr__(self, "weight", w if w > ZERO else ZERO)


@dataclass(frozen=True, slots=True)
class SentimentScore:
    """Aggregate sentiment for one interval, aligned to a bar's close.

    ``ts`` is the bucket CLOSE time, so it matches a :class:`~spintrader.core.types.Bar`
    ``ts`` exactly and a strategy can look it up by the current bar's timestamp.
    """
    ts: datetime
    symbol: str
    score: Decimal            # volume-weighted polarity, in [-1, 1]
    mentions: int             # how many raw mentions fell in the bucket
    volume: Decimal           # total engagement weight in the bucket
    source: str
    interval: str = "1m"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        object.__setattr__(self, "score", _clamp_score(to_decimal(self.score)))


# --------------------------------------------------------------------------
# Parsing -- adapted from tradingagents.dataflows.{stocktwits,reddit}
# --------------------------------------------------------------------------

def _parse_ts(value: Any) -> datetime | None:
    """Best-effort parse of a source timestamp into aware UTC.

    Accepts a unix epoch (Reddit's ``created_utc``) or an ISO-8601 string
    (StockTwits' ``created_at``). Returns ``None`` on anything unparseable, so a
    single malformed row is dropped rather than aborting the page.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    # StockTwits stamps look like "2026-07-29T12:34:56Z".
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_stocktwits_messages(
    symbol: str, messages: Iterable[Mapping[str, Any]],
) -> list[SentimentMention]:
    """Turn StockTwits stream messages into mentions.

    Mirrors :func:`tradingagents.dataflows.stocktwits.fetch_stocktwits_messages`
    field extraction: the user-supplied ``entities.sentiment.basic`` label is
    authoritative (``Bullish``/``Bearish``); an unlabelled message falls back to
    the keyword estimate over its body. Weight is one plus reshare/like counts,
    so a message the crowd amplified counts for more.
    """
    out: list[SentimentMention] = []
    for m in messages:
        if not isinstance(m, Mapping):
            continue
        ts = _parse_ts(m.get("created_at"))
        if ts is None:
            continue
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") if isinstance(entities, Mapping) else None
        label = sentiment_obj.get("basic") if isinstance(sentiment_obj, Mapping) else None
        body = str(m.get("body") or "")
        if label == "Bullish":
            score = ONE
        elif label == "Bearish":
            score = NEG_ONE
        else:
            score = score_text(body)
        likes = ((m.get("likes") or {}).get("total")
                 if isinstance(m.get("likes"), Mapping) else 0) or 0
        reshares = ((m.get("reshares") or {}).get("reshared_count")
                    if isinstance(m.get("reshares"), Mapping) else 0) or 0
        weight = ONE + to_decimal(likes) + to_decimal(reshares)
        out.append(SentimentMention(
            ts=ts, symbol=symbol, score=score, weight=weight, source="stocktwits",
        ))
    return out


def parse_reddit_posts(
    symbol: str, source: str, posts: Iterable[Mapping[str, Any]],
) -> list[SentimentMention]:
    """Turn Reddit search posts into mentions.

    Mirrors :func:`tradingagents.dataflows.reddit.fetch_reddit_posts` field
    extraction: ``created_utc``, ``score``, ``num_comments`` and the
    ``title``/``selftext`` text. Reddit does not label sentiment, so polarity is
    the keyword estimate over title-plus-body; weight is one plus the engagement
    magnitude (``|score| + num_comments``), which stands in for how much
    attention the post drew.
    """
    out: list[SentimentMention] = []
    for p in posts:
        if not isinstance(p, Mapping):
            continue
        ts = _parse_ts(p.get("created_utc"))
        if ts is None:
            continue
        title = str(p.get("title") or "")
        selftext = str(p.get("selftext") or "")
        polarity = score_text(f"{title} {selftext}")
        ups = abs(int(p.get("score") or 0))
        comments = int(p.get("num_comments") or 0)
        weight = ONE + to_decimal(ups) + to_decimal(comments)
        out.append(SentimentMention(
            ts=ts, symbol=symbol, score=polarity, weight=weight, source=source,
        ))
    return out


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

# A fetcher takes (symbol, since) and returns raw mentions. Mirrors the way
# kraken_trades' Fetcher mirrors fetch_trades, so a test injects a fake page
# source with no live endpoint and no mock HTTP session.
Fetcher = Callable[..., "list[SentimentMention]"]


def _base_ticker(symbol: str) -> str:
    """Reduce a canonical symbol to a bare ticker for a query.

    ``BTC-USD`` -> ``BTC``; ``AAPL`` -> ``AAPL``. The social endpoints key off a
    plain ticker, not SpinTrader's ``BASE-QUOTE`` form.
    """
    return symbol.split("-", 1)[0].upper()


def _default_fetcher(
    session: requests.Session,
    subreddits: Sequence[str] = DEFAULT_SUBREDDITS,
    timeout: float = 10.0,
    limit: int = 30,
) -> Fetcher:
    """Build a live fetcher over StockTwits + Reddit.

    Each source is isolated: a failure in one (network, HTTP-200-with-error,
    malformed JSON) logs and yields nothing rather than taking the other down or
    raising. This is the ``kraken_trades`` HTTP-200-with-error discipline applied
    to two keyless social endpoints -- the difference is that a missing social
    read is tolerable, so it degrades instead of aborting.
    """
    ticker = None

    def fetch(symbol: str, since: datetime | None = None) -> list[SentimentMention]:
        nonlocal ticker
        ticker = _base_ticker(symbol)
        mentions: list[SentimentMention] = []
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

        # -- StockTwits -----------------------------------------------------
        try:
            resp = session.get(
                STOCKTWITS_API.format(ticker=ticker), headers=headers, timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            # 200-with-error: StockTwits reports failure inside a 200 body.
            errors = payload.get("errors") if isinstance(payload, Mapping) else None
            if errors:
                log.warning("stocktwits %s: %s", ticker, errors)
            else:
                msgs = payload.get("messages", []) if isinstance(payload, Mapping) else []
                mentions += parse_stocktwits_messages(symbol, msgs[:limit])
        except (requests.RequestException, ValueError) as exc:
            log.warning("stocktwits fetch failed for %s: %s", ticker, exc)

        # -- Reddit ---------------------------------------------------------
        for sub in subreddits:
            try:
                resp = session.get(
                    REDDIT_API.format(sub=sub),
                    params={"q": ticker, "restrict_sr": "on", "sort": "new",
                            "t": "week", "limit": limit},
                    headers=headers, timeout=timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                children = ((payload.get("data") or {}).get("children") or []
                            if isinstance(payload, Mapping) else [])
                posts = [c.get("data", {}) for c in children if isinstance(c, Mapping)]
                mentions += parse_reddit_posts(symbol, f"reddit:{sub}", posts)
            except (requests.RequestException, ValueError) as exc:
                log.warning("reddit fetch failed for r/%s %s: %s", sub, ticker, exc)

        return mentions

    return fetch


def fetch_sentiment(
    symbol: str,
    since: datetime | None = None,
    fetcher: Fetcher | None = None,
    session: requests.Session | None = None,
) -> list[SentimentMention]:
    """Fetch raw mentions for ``symbol``, sorted ascending, filtered by ``since``.

    The network lives entirely behind ``fetcher``: pass one and no request is
    made, which is how the tests run offline. With no fetcher a live
    StockTwits+Reddit reader is built over ``session`` (or a fresh one). Mentions
    are returned ascending by time -- the order the aggregator requires -- and
    trimmed to those at or after ``since``.
    """
    fetch = fetcher or _default_fetcher(session or requests.Session())
    mentions = list(fetch(symbol, since))

    if since is not None:
        since_utc = ensure_utc(since)
        mentions = [m for m in mentions if m.ts >= since_utc]

    mentions.sort(key=lambda m: m.ts)
    return mentions


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

class SentimentBucketAggregator:
    """Buckets ascending mentions into per-interval sentiment, one bucket held.

    The subtle property, borrowed wholesale from
    :class:`~spintrader.data.kraken_trades.MinuteBarAggregator`: the most recent
    bucket is *held*, never emitted, until a mention in a later bucket proves it
    complete. That lets mentions arrive across many fetches -- a bucket split
    arbitrarily across a boundary -- and still produce exactly one score per
    interval whose ``ts`` is a bar close a strategy can already see.
    """

    def __init__(self, symbol: str, interval_minutes: int = 1) -> None:
        if interval_minutes < 1:
            raise ValueError("interval_minutes must be at least 1")
        self.symbol = symbol
        self.interval_minutes = interval_minutes
        self.interval = f"{interval_minutes}m"
        self._delta = timedelta(minutes=interval_minutes)
        # State for the bucket currently forming.
        self._start: datetime | None = None
        self._weighted_sum: Decimal = ZERO   # sum(score * weight)
        self._score_sum: Decimal = ZERO      # sum(score), for the zero-weight fallback
        self._weight: Decimal = ZERO         # sum(weight)
        self._count: int = 0
        self._sources: set[str] = set()

    def bucket_start(self, ts: datetime) -> datetime:
        """Floor ``ts`` to the start of its interval bucket.

        Computed off the wall-clock datetime, like the trades aggregator, so the
        boundary is exact: a mention at exactly 10:04:00 opens the 10:04 bucket,
        it does not close the 10:03 one.
        """
        ts = ensure_utc(ts)
        floored = ts.replace(second=0, microsecond=0)
        discard = floored.minute % self.interval_minutes
        return floored - timedelta(minutes=discard)

    def add(self, mentions: Iterable[SentimentMention]) -> list[SentimentScore]:
        """Fold mentions in and return every bucket they *complete*.

        The forming (latest) bucket is not returned -- only buckets proven done
        by a later mention. Mentions must be ascending; a regression signals a
        sorting error and is skipped rather than corrupting an emitted score.
        """
        completed: list[SentimentScore] = []
        for mention in mentions:
            start = self.bucket_start(mention.ts)
            if self._start is None:
                self._begin(start, mention)
            elif start == self._start:
                self._accumulate(mention)
            elif start > self._start:
                completed.append(self._finalise())
                self._begin(start, mention)
            else:
                log.warning(
                    "%s: out-of-order mention at %s (< current bucket %s); skipping",
                    self.symbol, mention.ts, self._start,
                )
        return completed

    def flush(self) -> SentimentScore | None:
        """Finalise and return the held bucket, if any, then clear it.

        Unlike a live price backfill, a sentiment series is usually built over a
        closed historical window, so the caller normally *does* want the final
        bucket. :meth:`SentimentFeed.series` flushes by default.
        """
        if self._start is None:
            return None
        return self._finalise()

    @property
    def has_open_bucket(self) -> bool:
        return self._start is not None

    @property
    def open_bucket_start(self) -> datetime | None:
        return self._start

    # -- internals ---------------------------------------------------------

    def _begin(self, start: datetime, mention: SentimentMention) -> None:
        self._start = start
        self._weighted_sum = mention.score * mention.weight
        self._score_sum = mention.score
        self._weight = mention.weight
        self._count = 1
        self._sources = {mention.source}

    def _accumulate(self, mention: SentimentMention) -> None:
        self._weighted_sum += mention.score * mention.weight
        self._score_sum += mention.score
        self._weight += mention.weight
        self._count += 1
        self._sources.add(mention.source)

    def _finalise(self) -> SentimentScore:
        assert self._start is not None
        if self._weight > ZERO:
            score = self._weighted_sum / self._weight
        else:
            # Every mention had zero engagement: fall back to a simple mean so a
            # freshly-posted, unamplified message still registers a polarity.
            score = self._score_sum / Decimal(self._count)
        source = next(iter(self._sources)) if len(self._sources) == 1 else "mixed"
        score_obj = SentimentScore(
            ts=self._start + self._delta,           # close-time convention
            symbol=self.symbol,
            score=_clamp_score(score),
            mentions=self._count,
            volume=self._weight,
            source=source,
            interval=self.interval,
        )
        self._start = None
        return score_obj


# --------------------------------------------------------------------------
# Feed
# --------------------------------------------------------------------------

class SentimentFeed:
    """Fetch-and-aggregate front door: raw mentions in, a bar-aligned series out.

    Network stays optional and behind the injectable fetcher, exactly as in
    :func:`fetch_sentiment`; a test constructs the feed with a fake fetcher and
    never touches a socket.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.session = session
        self.fetcher = fetcher

    def series(
        self,
        symbol: str,
        since: datetime | None = None,
        interval_minutes: int = 1,
        flush: bool = True,
    ) -> list[SentimentScore]:
        """Return the per-interval :class:`SentimentScore` series for ``symbol``.

        With ``flush=True`` (the default) the final, otherwise-held bucket is
        emitted too -- appropriate when building over a closed historical window.
        Pass ``flush=False`` to keep the live-edge discipline of the trades feed,
        where the forming bucket is dropped and re-acquired next run.
        """
        mentions = fetch_sentiment(
            symbol, since=since, fetcher=self.fetcher, session=self.session,
        )
        agg = SentimentBucketAggregator(symbol, interval_minutes)
        scores = agg.add(mentions)
        if flush:
            tail = agg.flush()
            if tail is not None:
                scores.append(tail)
        return scores

    def mapping(
        self,
        symbol: str,
        since: datetime | None = None,
        interval_minutes: int = 1,
        flush: bool = True,
    ) -> dict[datetime, SentimentScore]:
        """The series as a ``ts -> SentimentScore`` map, ready for a SentimentAgent.

        The agent is constructed with a sentiment mapping (bars do not carry
        sentiment), and its keys are bar close-times -- exactly the ``ts`` these
        scores are stamped with.
        """
        return {s.ts: s for s in self.series(symbol, since, interval_minutes, flush)}


__all__ = [
    "DEFAULT_SUBREDDITS", "Fetcher", "REDDIT_API", "STOCKTWITS_API",
    "SentimentBucketAggregator", "SentimentFeed", "SentimentMention",
    "SentimentScore", "USER_AGENT", "fetch_sentiment", "parse_reddit_posts",
    "parse_stocktwits_messages", "score_text",
]
