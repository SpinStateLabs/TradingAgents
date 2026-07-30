-- SpinTrader TimescaleDB schema.
--
-- Applied idempotently by spintrader.data.store.Store.migrate().
--
-- Design notes
-- ------------
-- * Bar timestamps are CLOSE times. Storing open-time is the single most
--   common source of off-by-one-bar lookahead, so the convention is fixed here
--   and asserted at ingestion.
-- * Prices are NUMERIC, not DOUBLE PRECISION. Float aggregates drift, and a
--   backtest that sums thousands of float P&L values diverges from the ledger
--   in ways that are tedious to chase.
-- * Every table that feeds a decision carries enough provenance to answer
--   "why did we do that" months later — which source, which model, which run.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- --------------------------------------------------------------------------
-- Instruments
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS instruments (
    instrument_key   TEXT PRIMARY KEY,          -- 'kraken:BTC-USD'
    venue            TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    venue_symbol     TEXT NOT NULL,
    asset_class      TEXT NOT NULL,
    base_currency    TEXT,
    quote_currency   TEXT NOT NULL,
    price_increment  NUMERIC NOT NULL DEFAULT 0.01,
    qty_increment    NUMERIC NOT NULL DEFAULT 0.00000001,
    min_qty          NUMERIC NOT NULL DEFAULT 0,
    min_notional     NUMERIC NOT NULL DEFAULT 0,
    maker_fee        NUMERIC NOT NULL DEFAULT 0,
    taker_fee        NUMERIC NOT NULL DEFAULT 0,
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (venue, symbol)
);

-- --------------------------------------------------------------------------
-- Bars
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bars (
    instrument_key TEXT        NOT NULL REFERENCES instruments(instrument_key),
    interval       TEXT        NOT NULL,        -- '1m','5m','1h','1d'
    ts             TIMESTAMPTZ NOT NULL,        -- bar CLOSE time, UTC
    open           NUMERIC     NOT NULL,
    high           NUMERIC     NOT NULL,
    low            NUMERIC     NOT NULL,
    close          NUMERIC     NOT NULL,
    volume         NUMERIC     NOT NULL DEFAULT 0,
    trades         INTEGER,
    vwap           NUMERIC,
    source         TEXT        NOT NULL,        -- 'kraken','yfinance','ibkr'
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instrument_key, interval, ts),
    CONSTRAINT bars_high_low CHECK (high >= low),
    CONSTRAINT bars_high_bounds CHECK (high >= open AND high >= close),
    CONSTRAINT bars_low_bounds  CHECK (low  <= open AND low  <= close)
);

SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS bars_instrument_interval_ts
    ON bars (instrument_key, interval, ts DESC);

-- --------------------------------------------------------------------------
-- Quotes (top of book snapshots, sampled)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS quotes (
    instrument_key TEXT        NOT NULL REFERENCES instruments(instrument_key),
    ts             TIMESTAMPTZ NOT NULL,
    bid            NUMERIC     NOT NULL,
    ask            NUMERIC     NOT NULL,
    bid_size       NUMERIC     NOT NULL DEFAULT 0,
    ask_size       NUMERIC     NOT NULL DEFAULT 0,
    source         TEXT        NOT NULL,
    PRIMARY KEY (instrument_key, ts)
);

SELECT create_hypertable('quotes', 'ts', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '1 day');

-- --------------------------------------------------------------------------
-- Orders and fills — the audit trail
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    venue_order_id  TEXT,
    client_order_id TEXT,
    decision_id     TEXT,                       -- links back to the reasoning
    strategy        TEXT,
    instrument_key  TEXT NOT NULL REFERENCES instruments(instrument_key),
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    qty             NUMERIC NOT NULL,
    limit_price     NUMERIC,
    stop_price      NUMERIC,
    time_in_force   TEXT NOT NULL,
    mode            TEXT NOT NULL,              -- backtest|paper|live
    status          TEXT NOT NULL,
    filled_qty      NUMERIC NOT NULL DEFAULT 0,
    avg_fill_price  NUMERIC,
    fees_paid       NUMERIC NOT NULL DEFAULT 0,
    reject_reason   TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    tags            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS orders_decision ON orders (decision_id);
CREATE INDEX IF NOT EXISTS orders_mode_created ON orders (mode, created_at DESC);

CREATE TABLE IF NOT EXISTS fills (
    fill_id        TEXT PRIMARY KEY,
    order_id       TEXT NOT NULL REFERENCES orders(order_id),
    venue_fill_id  TEXT,
    instrument_key TEXT NOT NULL REFERENCES instruments(instrument_key),
    side           TEXT NOT NULL,
    qty            NUMERIC NOT NULL,
    price          NUMERIC NOT NULL,
    fee            NUMERIC NOT NULL DEFAULT 0,
    fee_currency   TEXT NOT NULL DEFAULT 'USD',
    liquidity      TEXT,
    mode           TEXT NOT NULL,
    ts             TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS fills_order ON fills (order_id);
CREATE INDEX IF NOT EXISTS fills_instrument_ts ON fills (instrument_key, ts DESC);

-- --------------------------------------------------------------------------
-- Decisions — why the system did what it did
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS decisions (
    decision_id    TEXT PRIMARY KEY,
    instrument_key TEXT NOT NULL REFERENCES instruments(instrument_key),
    ts             TIMESTAMPTZ NOT NULL,
    action         TEXT NOT NULL,
    confidence     NUMERIC NOT NULL,
    target_weight  NUMERIC,
    horizon        TEXT NOT NULL DEFAULT '1d',
    regime         TEXT,
    rationale      TEXT,
    -- Per-agent votes and signal values. This is what makes post-hoc
    -- attribution possible: without it the self-improvement loop cannot tell
    -- which agent earned or lost the money.
    contributions  JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    mode           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS decisions_instrument_ts ON decisions (instrument_key, ts DESC);
CREATE INDEX IF NOT EXISTS decisions_mode_ts ON decisions (mode, ts DESC);

-- --------------------------------------------------------------------------
-- Regime states — output of the Markovian layer
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS regimes (
    instrument_key TEXT        NOT NULL REFERENCES instruments(instrument_key),
    ts             TIMESTAMPTZ NOT NULL,
    model          TEXT        NOT NULL,        -- 'gaussian_hmm_3state'
    model_version  TEXT        NOT NULL,
    state          INTEGER     NOT NULL,
    state_label    TEXT,                        -- 'bull_quiet','crisis',...
    -- Full posterior, not just the argmax: a 51/49 split between calm and
    -- crisis must not be acted on as though it were certainty.
    probabilities  JSONB       NOT NULL,
    risk_score     NUMERIC     NOT NULL,        -- 0 benign .. 1 crisis
    -- Fit window, so a state can be reproduced and checked for lookahead.
    fit_start      TIMESTAMPTZ NOT NULL,
    fit_end        TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (instrument_key, model, ts)
);

SELECT create_hypertable('regimes', 'ts', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '30 days');

-- --------------------------------------------------------------------------
-- Equity curve — one row per mark, per mode
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS equity_curve (
    ts             TIMESTAMPTZ NOT NULL,
    mode           TEXT        NOT NULL,
    run_id         TEXT        NOT NULL DEFAULT 'live',
    equity         NUMERIC     NOT NULL,
    cash           NUMERIC     NOT NULL,
    unrealized_pnl NUMERIC     NOT NULL DEFAULT 0,
    realized_pnl   NUMERIC     NOT NULL DEFAULT 0,
    fees_paid      NUMERIC     NOT NULL DEFAULT 0,
    gross_exposure NUMERIC     NOT NULL DEFAULT 0,
    positions      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (mode, run_id, ts)
);

SELECT create_hypertable('equity_curve', 'ts', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '30 days');

-- --------------------------------------------------------------------------
-- Research cache — "never download the same thing twice"
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS research_cache (
    cache_key    TEXT PRIMARY KEY,      -- stable hash of (source, query, window)
    source       TEXT NOT NULL,
    url          TEXT,
    content_hash TEXT NOT NULL,         -- detects changed content on refetch
    payload      JSONB NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ,
    hit_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS research_cache_source ON research_cache (source, fetched_at DESC);
CREATE INDEX IF NOT EXISTS research_cache_expiry ON research_cache (expires_at)
    WHERE expires_at IS NOT NULL;
