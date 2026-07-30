# Capital-Flows Watchlist — SpinTrader

**Purpose.** This is the *observability layer* for the regime allocator (3a), not an edge by itself. Price-based instruments **proxy** where capital is rotating (risk-on ↔ risk-off, growth ↔ value, cyclical ↔ defensive, US ↔ world, stocks ↔ bonds ↔ gold ↔ cash). Tracking the *ratios* between these tells you the market's current risk state, which then gates the allocation.

**Honest caveat (FORCE).** These are price proxies, **not literal fund-flow data** (ETF creation/redemption, CFTC positioning, EPFR, money-market assets). Price and flow usually agree, but not always. Real flow datasets are listed at the bottom as a follow-up. Not live-validated this session (sandbox couldn't reach Yahoo, GB10 Wi-Fi flaky) — all 70 are standard, highly-liquid, currently-trading instruments and get validated on first bar-fetch.

70 tickers, 15 groups. Machine-readable version: `flows_watchlist.csv`.

## The signals that matter — key ratio pairs

The individual assets are inputs; **the read is in the ratios.** Watch these:

- **Risk-on vs risk-off (equity):** `XLY / XLP` (consumer discretionary ÷ staples). Rising = risk-on.
- **Credit appetite:** `HYG / LQD` and `HYG / IEF` (junk vs quality vs Treasuries). Falling = credit stress — the earliest recession tell.
- **Breadth:** `RSP / SPY` (equal-weight ÷ cap-weight). Falling = rally narrowing to mega-caps (fragile).
- **Size appetite:** `IWM / SPY` (small ÷ large). Rising = broad risk-on.
- **Style rotation:** `IWF / IWD` (growth ÷ value). The dominant multi-year capital rotation.
- **Leadership:** `SMH / SPY` (semis ÷ market). Your AI-cycle tell.
- **Growth vs fear (macro):** `CPER / GLD` (copper ÷ gold — "Dr. Copper vs the panic metal"). Rising = growth optimism.
- **Global rotation:** `SPY / ACWX` (US ÷ world-ex-US) and `EEM / SPY` (EM risk).
- **Duration / flight-to-safety:** `TLT` direction and `^TNX` (10y yield). Bonds bid + yields falling into an equity selloff = flight to safety.
- **Fear:** `^VIX` level and `VIXY` (tradeable long-vol — your convexity/tail sleeve).
- **Dollar liquidity:** `UUP` (rising dollar drains global risk); `FXY`/`FXF` bid = risk-off haven demand.
- **Crypto as liquidity gauge:** `BTC-USD` leads risk appetite at the margin; `COIN`/`MSTR` are the equity expressions.

A simple **risk-on/off composite** for the regime allocator: average the z-scores of XLY/XLP, HYG/IEF, RSP/SPY, CPER/GLD (risk-on, positive) minus TLT, GLD, ^VIX, UUP (risk-off). That single series is what gates risk-on (AI-tech trend) vs risk-off (gold/Treasuries/cash + long-vol).

## The groups

- **Risk barometer (5):** SPY, QQQ, IWM, RSP, DIA — baseline equity risk appetite and breadth.
- **Credit (4):** HYG, JNK, LQD, EMB — the single most important non-equity risk read.
- **Rates (5):** TLT, IEF, SHY, TIP, BIL — duration, the curve, inflation expectations (TIP/IEF), and a true cash proxy (BIL).
- **Volatility (2):** ^VIX, VIXY — fear and the tradeable long-vol sleeve.
- **Sectors (11):** the full SPDR set (XLK/XLC/XLY/XLP/XLF/XLE/XLV/XLI/XLB/XLU/XLRE) — intra-equity rotation.
- **Theme (4):** SMH, SOXX, IGV, QTUM — semis/software/quantum leadership (your focus).
- **Factor (7):** IWF, IWD, MTUM, QUAL, USMV, SPLV, VLUE — style rotation and de-risking (USMV/SPLV bid = defense).
- **Safe haven (2):** GLD, SLV — monetary hedges.
- **FX (4):** UUP, FXY, FXF, FXE — dollar liquidity and haven currencies.
- **Commodity (7):** DBC, USO, UNG, CPER, DBA, URA, XME — inflation and growth demand.
- **Global (10):** EEM, VWO, EFA, VGK, FXI, MCHI, EWJ, INDA, EWZ, ACWX — geographic flows.
- **Crypto (3) + proxies (2):** BTC/ETH/SOL-USD, COIN, MSTR — digital liquidity.
- **Hedge (2):** BTAL (anti-beta), DBMF (managed futures / crisis alpha).
- **Yield (2):** ^TNX, ^IRX — rate direction and policy proxy (levels, not tradeable here).

## Ingestion

Fetchable free via the existing `scripts/fetch_bars_csv.py` (yfinance, daily). Equities/ETFs/FX-ETFs and `^`-index symbols all resolve; crypto is already in the store. On the GB10:
`for t in <tickers>; do python scripts/fetch_bars_csv.py $t --years 15 --out data/flows/$t.csv; done` — or backfill into TimescaleDB. Then the regime allocator computes the ratios/z-scores above each day.

## Follow-up: real capital-flow data (beyond price proxies)
- **CFTC Commitments of Traders** (free) — futures positioning by trader type (the truest "who's buying" for indices, rates, FX, commodities).
- **ETF fund flows** (etf.com, issuer sites; some free) — actual creation/redemption dollars.
- **ICI money-market fund assets & Fed H.4.1 / reserves** (free) — cash-on-sidelines and system liquidity.
- **EPFR / Lipper** (paid) — global institutional equity/bond fund flows.
- **On-chain** (Glassnode/CryptoQuant; free tiers) — exchange in/outflows, stablecoin supply for the crypto sleeve.
- **Options/dark-pool** (paid) — dealer positioning; heavier lift, later.
These are where "profit from news/flows" gets its real signal; wire them after the price-proxy layer is proven.
