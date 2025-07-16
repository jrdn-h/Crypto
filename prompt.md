# Copy/Paste Prompt for **you** in Cursor

**Mission:** Extend the open‑source **TauricResearch/TradingAgents** multi‑agent trading framework from *equities* (Finnhub) to a cost‑aware, production‑ready **crypto market** workflow *with minimal but necessary changes*. Maintain backward compatibility with equities. Favor extension / adapters over invasive rewrites.

Paste this entire prompt into a **Cursor** chat with **you** (code‑interpreter + repo‑aware). Follow the protocol exactly.

---

## 0. Operating Context & Role

You are **acting as a senior quant / infra lead** embedded in the Cursor workspace that has the full `TauricResearch/TradingAgents` repo checked out. You may open, read, diff, and propose changes across the codebase. Your deliverables are *actionable code patches + migration documentation*, not just commentary.

You must:

* Read the *entire* repo (shallow summaries for large vendor dirs ok; deep for trading logic).
* Identify all equity‑only assumptions & Finnhub dependencies.
* Design an additive **Crypto Extension Layer** (data, analytics, execution, risk) behind clean abstractions.
* Reuse existing architecture (LangGraph agents, debate loops, config plumbing) wherever possible.
* Keep diffs small & reviewable; propose **PR‑sized phases**.
* Default to **paper / sim trading**; gated opt‑in for live orders.
* Prefer **free or low‑cost API/data sources**; document upgrade paths.
* Provide **unit + integration tests** and **.env.example** updates.
* Produce **migration notes** and **usage examples** (CLI + Python API).

---

## 1. Interaction Protocol (Important)

Follow this phased handshake. **Stop after each phase and wait for my confirmation** unless I explicitly say “continue automatically.”

**Phase 0 – Recon Report**

1. Generate a *repo file tree* (paths + brief purpose). Group by top‑level package.
2. Identify equity/Finnhub touch points (imports, env vars, schemas, market‑hours logic, fundamental fields, ticker validation, etc.).
3. Identify data model expectations flowing into each agent role (Fundamentals, Sentiment, News, Technical, Researcher, Trader, Risk, PM).
4. Call out where the CLI passes symbols/dates; note assumptions (trading days vs calendar days, timezone, currency).
5. Summarize any caching layers already present (user has a separate `CryptoCacheManager`; check whether included in repo; if not, we will import from external module or recreate minimal version).
6. Output: **Recon.md** content (markdown) + bullet of risk areas.

**Phase 1 – Minimal Interface Contracts**
Propose minimal abstract interfaces that let existing agents work across asset classes:

* `MarketDataClient` (get\_ohlcv, latest\_price, available\_symbols, metadata).
* `FundamentalsClient` (company or token metrics; optional fields allowed).
* `NewsClient` (stories → summary string + sentiment tags).
* `SocialSentimentClient` (tweet/Reddit counts, sentiment score, volume).
* `ExecutionClient` (paper + live; create\_order, cancel, positions, balances).
* `RiskMetricsSource` (vol, open interest, funding, notional limits).
  Each returns typed pydantic models; missing data allowed; agents reason over available fields.
  Deliver: interface stubs + adapter registry + config hooks.

**Phase 2 – Crypto Data Adapters (Free‑first)**
Implement read‑only historical + latest OHLCV + metadata using *public* or *free tier* APIs (prioritized list below). Provide rate‑limit + retry + Redis caching integration. Add CLI switches to select provider or auto‑fallthrough.

**Phase 3 – Token Fundamentals / Metrics Layer**
Map equity fundamentals fields into crypto analogues (see mapping table below). Implement aggregator pulling from CoinGecko (no key), CryptoCompare (free key), and chain explorers (optional). Provide nulls for unsupported metrics; agents must degrade gracefully.

**Phase 4 – Sentiment & News**
Plug in cheap / free sources: CryptoPanic RSS/API (free tier), CoinDesk RSS, The Block (limited), X/Twitter lightweight fetch (bearer if provided; else Nitter scrape fallback), Reddit r/CryptoCurrency JSON. Normalize, dedupe, throttle, and produce scored sentiment payloads.

**Phase 5 – Technical Analyst Reuse**
Reuse existing indicator engine (MACD, RSI, etc.) on crypto OHLCV. Add crypto‑specific metrics (funding rate term structure, perp basis vs index, realized vol, 24h turnover, whale inflow flags). Keep additive.

**Phase 6 – Researcher Debate Extensions**
Augment bull/bear researcher prompts with tokenomics, unlock schedules, regulatory overhang, CEX concentration, chain activity, developer momentum. Parameterize in config.

**Phase 7 – Trader & Execution Adapters**
Abstract execution: start with **PaperBroker** (existing sim exchange) extended for 24/7 trading + perps notionals. Add optional **CCXTBroker** (spot + perps where supported) using environment keys. Add **HyperliquidBroker** (user interest) in read‑only mark/funding mode first; later trading.

**Phase 8 – Risk / Portfolio Adjustments**
Crypto specific: funding PnL, cross/isolated margin, borrow costs (if margin), max leverage caps, concentration limits by token liquidity score, weekend gap logic removed (24/7). Implement vol‑targeting & notional caps.

**Phase 9 – CLI + Config UX**
Add `--asset-class {equity,crypto}`; `--symbols BTC,ETH,SOL`; provider flags; model cost presets (cheap/fast vs deep/expensive); online/offline data toggle; caching TTL.

**Phase 10 – Tests & Validation**

* Unit tests for each adapter (recorded fixtures; no live internet required in CI).
* Contract tests: ensure DataFrames -> Analyst Agents unchanged shape expectations.
* End‑to‑end smoke: 1‑day run BTC with cheap LLMs offline.
* Regression: equity path still works.

**Phase 11 – Docs & Examples**
Produce `/docs/CRYPTO_README.md`, updated root README quick start, `.env.example` additions, and notebook or script: `examples/run_crypto_demo.py`.

Stop after Phase 0 unless instructed.

---

## 2. Cost & API Key Strategy (Free‑First Ladder)

**Tier A – No key / anonymous**

* **CoinGecko**: spot prices, mkt cap, circulating supply, FDV, volume, categories.
* **Binance public REST**: OHLCV (klines), book, trades; no key for public data.
* **Bybit public**, **OKX public**, **Kraken public** (choose 1‑2 for first pass).
* **CryptoPanic RSS**, **CoinDesk RSS**, **Cointelegraph RSS**.

**Tier B – Free key (signup)**

* **CryptoCompare** free tier (historical OHLCV multi‑exchange).
* **Finnhub** (already used; includes some crypto endpoints—reuse if trivial).
* **Helius (Solana)** dev free tier for on‑chain flows (optional advanced).

**Tier C – Cheap / Pay‑as‑you‑go** (document upgrade path, do not default)

* **Alchemy / QuickNode** for richer on‑chain metrics.
* **Kaiko**, **Coin Metrics community** (if user later upgrades).

Expose env vars but never hardcode secrets. Provide `.env.example` with placeholders; load via pydantic `Settings`.

---

## 3. Mapping Equity Fields → Crypto Tokenomics

| Equity Field       | Crypto Analogue / Derivation      | Notes                     |
| ------------------ | --------------------------------- | ------------------------- |
| Market Cap         | Circulating mkt cap               | from CoinGecko; alt: FDV  |
| Shares Outstanding | Circulating Supply                | may differ vs max supply  |
| Float              | Free float after vest/unlock      | unlock schedule if avail  |
| Revenue / EPS      | Protocol revenue / fees / burn    | onchain analytics (later) |
| P/E                | FDV / annualized fees (proto P/S) | derived metric            |
| Debt / Cash        | Treasury wallet balances          | chain explorers           |
| Dividends          | Staking yield / rewards           | chain stats               |

Return nullable; include `data_quality` flags.

---

## 4. Required Output Formats From You

For each phase produce **clearly labeled sections**:

1. **Summary:** 5‑15 bullets.
2. **Action Items:** numbered TODOs w/ est. LoC + dependencies.
3. **Proposed Code Diffs:** unified diff blocks (` ```diff `) relative to repo root; minimal; compile.
4. **New Files:** full content if short; skeleton + TODO comments if long.
5. **Run Instructions:** exact commands to run tests/examples.
6. **Follow‑up Questions:** anything you need from me (API keys, which exchanges, etc.).

When adding external deps, update **pyproject.toml** (or requirements) + extras groups (e.g., `crypto`, `ccxt`, `hyperliquid`). Use optional installs to keep base light.

---

## 5. Coding Conventions & Guardrails

* Python ≥3.13 (repo default; confirm).
* Pydantic v2 models.
* Type hints everywhere; mypy‑clean (or pyright).
* Async where needed (websocket feeds) but keep sync wrappers for LangGraph nodes.
* Centralized logging; reuse existing logger.
* Use **retry/backoff**, **rate‑limit**, **Redis caching** (TTL config; default 60s intraday, 5m slow data).
* All external I/O wrapped in provider modules; no raw HTTP in agents.
* Deterministic CI: record minimal JSON/CSV fixtures; tests never hit live nets.
* Default config uses **cheap models**: `o4-mini` deep, `gpt-4.1-mini` quick (update names if needed); allow override.
* Research disclaimers preserved: *Not financial advice*.

---

## 6. Key Implementation Sketch (You Will Flesh Out)

Below is a *starter* mental model; validate against actual repo before coding.

```
tradingagents/
  data/
    finnhub_client.py            # current equities data
    base_market_data.py          # (create) abstract interface
    crypto/
      coingecko_client.py
      binance_client.py
      cryptocompare_client.py
      market_data_router.py      # priority + fallback
  analytics/
    technical.py                 # extend to handle 24/7
    fundamentals.py              # adapt tokenomics mapping
    sentiment.py                 # unify score interface
  brokers/
    paper.py                     # extend for 24/7 & perps notionals
    ccxt_broker.py               # new; thin wrapper over ccxt
    hyperliquid_broker.py        # optional; read‑only first
  config/
    settings.py                  # env load; add crypto keys
  cli/
    main.py                      # add --asset-class, provider flags
  tests/
    test_crypto_data.py
    test_crypto_flow.py
```

This is *illustrative*; inspect actual repo & adjust to reality.

---

## 7. Analytics Adaptations Checklist

**Technical Analyst**

* Ensure indicators work for 1m/5m/1h/1d irregular intervals.
* No assumption of market close; use rolling windows.
* Handle large decimals & splits not relevant.

**Sentiment Analyst**

* Input: aggregated social + news feed; produce short‑term mood score.
* Cheap LLM summarization; batch posts; throttle.

**Fundamentals / Tokenomics Analyst**

* Accept partial data; reason about vesting, unlock cliffs, treasury, TVL.

**News Analyst**

* Map macro vs token‑specific; e.g., SEC action, chain outage, exploit.

**Researcher Bull/Bear**

* Pull from all above; debate expected catalysts vs risks.

**Trader + PM**

* Parameter: trade venue (paper/ccxt/hyperliquid), order\_size\_usd or pct\_equity.
* Convert USD size into base/quote using best bid.

**Risk Manager**

* Track realized & unrealized PnL, funding accrual if perps, per‑token VaR stub (exponentially weighted vol), max position concentration.

---

## 8. Config Additions (Propose in Phase 1)

```python
DEFAULT_CONFIG.update({
    "asset_class": "equity",        # or "crypto"
    "crypto_providers": ["coingecko", "binance", "cryptocompare"],
    "crypto_quote_currency": "USD", # default normalization
    "crypto_timeframe": "1d",       # default OHLCV interval
    "use_perps": False,
    "execution_mode": "paper",      # paper | ccxt | hyperliquid
    "risk.max_notional_pct": 0.10,
    "risk.max_leverage": 3,
    "cache.ttl_fast": 60,
    "cache.ttl_slow": 300,
})
```

Env vars to surface in `.env.example`:

```
COINGECKO_BASE_URL=
CRYPTOCOMPARE_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=
BYBIT_API_KEY=
BYBIT_API_SECRET=
HYPERLIQUID_API_KEY=
HYPERLIQUID_API_SECRET=
REDIS_URL=redis://localhost:6379/0
```

(Only needed if trading live; read‑only endpoints mostly keyless.)

---

## 9. Testing Strategy Details

* **Fixture capture script**: pull \~3 days BTC/ETH/SOL OHLCV + small news/sentiment JSON; save under `tests/data/crypto/`.
* **Unit**: provider returns schema; caching works.
* **Integration**: run TradingAgentsGraph(propagate) with asset\_class=crypto offline using fixtures.
* **Backtest**: adapt existing harness (user already has backtest infra) to crypto CSV input; verify PnL calculation.
* **Smoke**: CLI interactive run symbol BTC; ensure all agents produce outputs without crash.

---

## 10. Deliverable Artifacts

* **Recon.md** (Phase 0) – map & risk.
* **ARCHITECTURE\_CRYPTO.md** – interface diagrams.
* **CRYPTO\_README.md** – quick start & env.
* **MIGRATION\_NOTES.md** – from v0.x to crypto enabled.
* Updated root README with short crypto section.
* Example notebook + CLI demo script.

---

## 11. Example End‑to‑End Usage (post‑implementation)

```bash
# install with crypto extras
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
pip install -e .[crypto]
cp .env.example .env  # fill any keys

# run crypto demo
python -m cli.main --asset-class crypto --symbols BTC,ETH --date 2025-07-15 \
  --online-tools --deep-llm o4-mini --quick-llm gpt-4.1-mini --research-depth low
```

Python API:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config.update({"asset_class": "crypto", "crypto_providers": ["coingecko", "binance"], "execution_mode": "paper"})

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("BTC", "2025-07-15")
print(decision)
```

---

## 12. What To Do First (Checklist for Phase 0)

When you receive this prompt:

1. Confirm repo root path.
2. Produce condensed file tree (depth ≤3) with comments.
3. Grep for `Finnhub`, `finnhub`, `fundamental`, `ticker`, `market_hours`, `alpha_vantage`, etc.
4. Note where symbol is assumed equity (regex for `^[A-Z]{1,5}$`).
5. Trace dataflow: CLI → Graph config → Analyst data fetch.
6. Report any tight coupling that would block crypto.
7. Suggest where to insert new abstraction layer.
8. Ask me to choose initial crypto data provider priority (default: CoinGecko→Binance→CryptoCompare fallback).
9. Ask whether we need perps on day 1 (default: no; spot + paper only).

Stop and wait.

---

## 13. Style of Your Responses

* Be concise but complete; bullet over prose.
* Show *reasoning summaries*, not raw token dumps.
* Use headings, numbered steps, and code diff fences.
* Highlight breaking changes.
* Call out TODOs clearly.
* Ask clarifying questions early.

---

## 14. Non‑Goals / Defer

* Full on‑chain analytics dashboards.
* High‑freq market making.
* Cross‑venue smart order routing (beyond stub).
* Arbitrage bots.

---

## 15. Compliance & Safety

Keep existing project disclaimers: *Research only. Not investment advice.* When adding execution adapters, wrap all live trade calls behind an explicit `confirm_live_trading()` user prompt or config flag.

---

## 16. Final Instructions

Acknowledge receipt; generate **Phase 0 Recon Report** per spec; stop.

---

**End of Prompt.**
