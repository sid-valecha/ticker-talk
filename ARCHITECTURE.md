# Ticker-Talk Architecture

## System Overview

Ticker-Talk is a natural-language stock analysis application that demonstrates AI-assisted financial data analysis. The system is designed with a clear separation of concerns: **Python computes all numbers deterministically**, and **LLMs only explain results**.

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│          Next.js Frontend               │
│  - Query form (ticker or CSV)           │
│  - Chart display (base64 PNGs)          │
│  - Explanation panel                    │
└──────────────┬──────────────────────────┘
               │ HTTP POST
               ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│                                         │
│  ┌────────────────────────────────┐    │
│  │  /api/analyze endpoint         │    │
│  │  1. Check SQLite cache         │    │
│  │  2. Fetch if needed (AV API)   │    │
│  │  3. Compute indicators         │    │
│  │  4. Run ARIMA forecast         │    │
│  │  5. Backtest with walk-forward │    │
│  │  6. Generate matplotlib plots  │    │
│  │  7. Call LLM for explanation   │    │
│  │  8. Log request metrics        │    │
│  │  9. Return JSON response       │    │
│  └────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
       │               │              │
       ▼               ▼              ▼
┌──────────┐  ┌─────────────┐  ┌──────────┐
│ SQLite   │  │ Alpha       │  │ OpenAI/  │
│ Cache    │  │ Vantage     │  │ Groq API │
│ (24h TTL)│  │ API         │  │          │
└──────────┘  └─────────────┘  └──────────┘
```

## Core Principles

### 1. Deterministic Computation
- **All math happens in Python** using pandas, numpy, statsmodels
- Indicators (returns, MA, RSI, volatility) computed with standard formulas
- ARIMA forecasting uses statsmodels (statistical model, not ML)
- Backtesting uses walk-forward validation with sklearn metrics
- **No LLM involvement in computation**

### 2. LLM for Reasoning Only
- LLMs parse natural language queries into structured requests (optional)
- LLMs generate explanations **grounded in computed results**
- LLMs never compute, predict, or generate numbers
- Strict guardrails prevent trading advice or ungrounded claims

### 3. Aggressive Caching
- Alpha Vantage free tier: **25 requests/day** (hard limit)
- SQLite cache with 24-hour TTL per ticker
- Cache key: ticker symbol
- Cache stores: raw OHLCV data + metadata
- Demo mode with preloaded tickers (AAPL, MSFT, GOOGL) as fallback

### 4. Self-Contained Deployment
- No cloud storage for plots (base64 PNG inline)
- No database server (SQLite file)
- No message queues or complex infra
- Single backend process + static frontend

## Data Flow

### Stock Analysis Request
```
1. User submits: {"ticker": "AAPL", "forecast_horizon": 7}

2. Cache check:
   - Query SQLite for ticker + timestamp
   - If hit and < 24h old → use cached data
   - If miss or stale → fetch from Alpha Vantage
   - If API rate limited → fallback to demo data or fail

3. Data processing:
   - Load OHLCV data into pandas DataFrame
   - Compute technical indicators:
     * returns = price.pct_change()
     * ma_20 = price.rolling(20).mean()
     * volatility = returns.rolling(20).std()
     * rsi = relative_strength_index(price, 14)

4. Forecasting (if requested):
   - Fit ARIMA(5,1,0) model on price series
   - Generate forecast for horizon (7 or 30 days)
   - Compute 95% confidence interval
   - Run walk-forward backtest (5 splits)
   - Calculate MAE, RMSE, MAPE

5. Plotting:
   - matplotlib: price + MA overlay
   - matplotlib: returns + volatility dual-axis
   - matplotlib: forecast + confidence band
   - Convert to PNG → base64 encode

6. LLM explanation:
   - Build summary dict: {indicators, forecast, backtest}
   - Call OpenAI/Groq with strict system prompt
   - Prompt includes guardrails (no advice, acknowledge uncertainty)
   - Return explanation text

7. Response assembly:
   - metadata: {cache_hit, data_last_updated, source, row_count}
   - indicators: {latest_return, ma_20, volatility, rsi, averages}
   - forecast: {dates, values, lower_ci, upper_ci}
   - backtest: {mae, rmse, mape, predictions, actuals}
   - plots: {price_ma_b64, returns_vol_b64, forecast_b64}
   - explanation: "..." (LLM-generated text)

8. Metrics logging:
   - Insert into request_metrics table
   - Record: endpoint, ticker, cache_hit, latency_ms
```

### CSV Upload Request
```
1. User uploads CSV file

2. Validation:
   - Check for required columns: date, price (or close/adj_close)
   - Parse date column to datetime
   - Validate data types and ranges

3. Normalization:
   - Rename price column to adj_close
   - Set date as index
   - Fill missing OHLCV columns with price if needed
   - Sort by date ascending

4. Cache storage:
   - Generate hash from file content
   - Store in SQLite with hash as key
   - Set TTL to 24h

5. Analysis pipeline:
   - Same as stock analysis (steps 3-7 above)
   - Source field = "csv_upload"
```

## Component Architecture

### Backend Modules

#### `app/data/` - Data Layer
- **cache.py:** SQLite operations (get, store, log metrics)
- **alpha_vantage.py:** API client for TIME_SERIES_DAILY_ADJUSTED
- **csv_loader.py:** CSV validation and normalization
- **demo_data.py:** Preloaded ticker data for offline mode

#### `app/compute/` - Computation Layer
- **indicators.py:** Technical indicator calculations (pure functions)
- **forecast_arima.py:** ARIMA model fitting and forecasting
- **backtest.py:** Walk-forward validation and metrics

#### `app/plots/` - Visualization Layer
- **charts.py:** matplotlib chart generation → base64 PNG

#### `app/llm/` - LLM Integration
- **client.py:** OpenAI/Groq client abstraction (provider-agnostic)
- **explain.py:** Explanation generation with guardrails

#### `app/api/` - API Endpoints
- **analyze.py:** POST /api/analyze (stock ticker analysis)
- **analyze_csv.py:** POST /api/analyze_csv (CSV upload analysis)
- **metrics.py:** GET /api/metrics (cache stats, usage metrics)

#### `app/models/` - Data Models
- **schemas.py:** Pydantic request/response models

### Frontend Components

#### `app/components/`
- **QueryForm.tsx:** Input form (ticker or CSV upload + options)
- **ChartsPanel.tsx:** Displays base64 image charts
- **ExplanationPanel.tsx:** Shows LLM explanation + metadata

#### `app/lib/`
- **api.ts:** Typed API client (fetch wrappers)

## Technology Choices

### Backend Stack
- **FastAPI:** Modern Python web framework, automatic OpenAPI docs, async support
- **SQLite:** Zero-config database, perfect for caching and metrics
- **pandas/numpy:** Industry-standard data manipulation
- **statsmodels:** Statistical forecasting (ARIMA)
- **matplotlib:** Chart generation (ubiquitous, reliable)
- **OpenAI SDK:** Compatible with both OpenAI and Groq APIs

**Why not:**
- ❌ PostgreSQL/Redis: Overkill for MVP, adds deployment complexity
- ❌ Prophet: Requires more data, slower, heavier dependency
- ❌ Plotly: Requires JSON transport + frontend rendering, adds complexity

### Frontend Stack
- **Next.js 14 (App Router):** Modern React framework, TypeScript support
- **Tailwind CSS:** Utility-first styling, fast iteration
- **No charting library:** Display base64 images directly (simpler)

**Why not:**
- ❌ Recharts/Plotly: Backend generates charts, frontend just displays
- ❌ Complex state management: Single request/response, no need for Redux

### LLM Integration
- **OpenAI API (gpt-4o-mini):** Fast, cheap, good structured output
- **Groq API (llama models):** Free tier, very fast inference, same SDK
- **Provider abstraction:** Easy to switch between OpenAI/Groq

**Why not:**
- ❌ Local models: Adds deployment complexity, slower
- ❌ Custom orchestration layer: Overkill for simple intent + explanation

## Database Schema

### SQLite Tables

#### `ticker_cache`
```sql
CREATE TABLE ticker_cache (
    ticker TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,        -- JSON array of OHLCV records
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count INTEGER,
    min_date TEXT,                   -- Earliest data point
    max_date TEXT                    -- Latest data point
);
```

**Purpose:** Cache Alpha Vantage responses to minimize API calls

**TTL:** 24 hours (checked on read)

**Size estimate:** ~50KB per ticker, 100 tickers = 5MB

#### `request_metrics`
```sql
CREATE TABLE request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT,                   -- "/api/analyze" or "/api/analyze_csv"
    ticker TEXT,                     -- Ticker symbol or "csv_upload"
    cache_hit BOOLEAN,               -- True if served from cache
    latency_ms INTEGER,              -- Request processing time
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose:** Track usage for metrics reporting

**Queries:**
- Total requests: `SELECT COUNT(*) FROM request_metrics`
- Cache hit rate: `SELECT AVG(cache_hit) FROM request_metrics`
- Top tickers: `SELECT ticker, COUNT(*) FROM request_metrics GROUP BY ticker ORDER BY COUNT(*) DESC LIMIT 10`

## Caching Strategy

### Cache Lifecycle
1. **Write:** After successful Alpha Vantage fetch
2. **Read:** Before every API call (check ticker + timestamp)
3. **Invalidation:** Time-based (24h TTL), checked on read
4. **Eviction:** Manual (future: LRU if storage grows)

### Cache Hit Optimization
- Preload popular tickers (AAPL, MSFT, GOOGL) on server start
- Demo mode serves from cache without API calls
- Daily data doesn't change intraday (safe to cache aggressively)

### Rate Limit Protection
- Max 25 Alpha Vantage calls/day (free tier)
- Cache reduces calls to ~1 per ticker per day
- If rate limited (HTTP 429 or API message), return cached data or fail gracefully
- Metrics track API call count (should be < 25/day in production)

## LLM Integration Details

### Intent Parsing (Optional)
**Input:** Natural language query
**Output:** Structured request dict

**Example:**
```
User: "forecast AAPL for 30 days"
→ {"ticker": "AAPL", "forecast_horizon": 30}

User: "show me Microsoft stock with 7-day prediction"
→ {"ticker": "MSFT", "forecast_horizon": 7}
```

**System Prompt:**
```
Extract stock analysis parameters from user query.
Return JSON with: ticker, forecast_horizon (7 or 30 or null).
Only valid horizons are 7 or 30 days.
If no horizon mentioned, set to null.
```

**Note:** MVP may skip this and use structured form inputs directly.

### Explanation Generation (Required)
**Input:** Computed analysis summary (JSON)
**Output:** 2-3 paragraph explanation

**System Prompt:**
```
You are a financial analysis assistant.

STRICT RULES:
- Base explanation ONLY on provided computed metrics
- Never give trading advice or recommendations
- Acknowledge uncertainty and limitations
- Explain what the numbers mean, not what to do
- If forecast exists, explain it's a model prediction with uncertainty
```

**Guardrails:**
- Never say "buy" or "sell"
- Never predict future performance as fact
- Always mention model limitations
- Cite specific metrics from input

**Example Output:**
```
Apple (AAPL) shows moderate volatility with a 20-day moving average
of $175.32, slightly above the current price of $174.85. Daily returns
average 0.12% over the last 30 days with a standard deviation of 1.8%.
The RSI of 58 suggests neutral momentum.

The ARIMA forecast predicts a gradual uptrend over the next 7 days,
with prices ranging between $173-$178 (95% confidence interval). However,
this is a statistical model with inherent uncertainty. Walk-forward
backtesting shows a mean absolute error of $2.15, indicating the model's
predictions typically deviate by about $2 from actual prices.

These metrics reflect historical patterns and should not be interpreted
as investment advice. Market conditions can change rapidly.
```

## Error Handling

### API Failures
- **Alpha Vantage down:** Serve cached data or demo mode
- **Rate limited:** Log error, serve cached or fail with 429
- **Invalid ticker:** Return 400 with clear error message
- **LLM API down:** Return analysis without explanation (optional field)

### Data Quality
- **Short time series:** Warn user if < 60 days data (ARIMA needs history)
- **Missing data:** Forward-fill gaps in price series
- **Extreme volatility:** Flag in metadata if volatility > 10%

### Validation
- **Forecast horizon:** Must be 7 or 30 (reject others with 400)
- **CSV schema:** Require date + price columns (reject malformed)
- **Date ranges:** Validate start_date < end_date
- **Ticker format:** Basic validation (1-5 uppercase letters)

## Performance Considerations

### Backend
- **Cold start:** ~2s (load models, init cache)
- **Cache hit:** ~200ms (compute + plot + LLM)
- **Cache miss:** ~3s (API fetch + compute + plot + LLM)
- **CSV upload:** ~500ms + compute time (depends on file size)

### Optimization Opportunities
- Use async HTTP clients (httpx) for Alpha Vantage
- Cache LLM explanations for identical analysis results
- Pre-compute indicators for cached data
- Batch plot generation if multiple charts

### Bottlenecks
- **Alpha Vantage API:** 25 calls/day limit (mitigated by cache)
- **LLM latency:** 500-1000ms (acceptable for MVP)
- **ARIMA fitting:** ~200ms for 1-year series (one-time per request)

## Security Considerations

### API Keys
- Store in environment variables (never commit)
- Validate presence on server start
- Rotate periodically

### Input Validation
- Sanitize ticker input (prevent SQL injection, though unlikely)
- Limit CSV file size (max 10MB)
- Rate limit frontend requests (future: 10 req/min per IP)

### Data Privacy
- No user accounts or stored queries
- Metrics are aggregated (no PII)
- CSV files not persisted beyond cache TTL

## Deployment Architecture

### Railway (Backend)
- **Environment:** Python 3.11
- **Process:** Single uvicorn worker
- **Storage:** SQLite file (ephemeral or attached volume)
- **Scaling:** Vertical only (1 instance, no horizontal scaling needed)

### Vercel (Frontend)
- **Environment:** Node.js 20
- **Build:** Static generation + API routes
- **Edge:** Global CDN for static assets
- **CORS:** Backend allows Vercel domain origin

### Environment Variables
**Backend:**
- `ALPHA_VANTAGE_API_KEY` (required)
- `OPENAI_API_KEY` or `GROQ_API_KEY` (required)
- `LLM_PROVIDER` (openai or groq)
- `LLM_MODEL` (gpt-4o-mini or llama model)
- `ENVIRONMENT` (production)

**Frontend:**
- `NEXT_PUBLIC_API_URL` (Railway backend URL)

## Future Enhancements (Out of MVP Scope)

### Features
- [ ] Model comparison (ARIMA vs Prophet side-by-side)
- [ ] Multiple ticker comparison charts
- [ ] Custom indicator formulas (user-defined TA-Lib)
- [ ] Email alerts for price thresholds
- [ ] Export analysis to PDF

### Infrastructure
- [ ] Redis cache for distributed deployment
- [ ] PostgreSQL for persistent metrics
- [ ] Celery for async forecast jobs
- [ ] Prometheus metrics + Grafana dashboards
- [ ] CI/CD with GitHub Actions

### LLM
- [ ] Multi-step agent for complex queries
- [ ] Vector DB for RAG over financial docs
- [ ] Fine-tuned model for financial explanations

## Metrics to Track

### Operational
- **Total requests:** Count of all API calls
- **Cache hit rate:** % of requests served from cache
- **Average latency:** Mean request processing time
- **API call count:** Alpha Vantage calls (should be < 25/day)
- **Error rate:** % of requests resulting in 4xx/5xx

### Usage
- **Top tickers:** Most frequently queried stocks
- **Forecast usage:** % of requests with forecast_horizon set
- **CSV uploads:** Count of CSV analysis requests
- **Peak hours:** Request distribution by hour

### Quality
- **Forecast accuracy:** Aggregated MAE/RMSE across backtests
- **LLM latency:** Time to generate explanations
- **Explanation length:** Avg word count (should be 100-300)

**Capture in README after 1-2 weeks:**
```
Performance Metrics (as of YYYY-MM-DD):
- Total queries: 487
- Cache hit rate: 73%
- Avg latency: 1.2s (cached), 3.5s (uncached)
- Alpha Vantage calls: 18/25 daily limit
- Most queried: AAPL (143), MSFT (89), GOOGL (67)
- CSV uploads: 23
```

## Design Decisions Log

### Why ARIMA over Prophet?
- Simpler dependency (statsmodels vs fbprophet)
- Faster training (~200ms vs ~2s)
- Good enough for 7-30 day horizons
- Prophet better for longer horizons (not MVP scope)

### Why SQLite over PostgreSQL?
- Zero deployment complexity
- Perfect for caching and metrics (< 1GB data)
- Can migrate to Postgres if scale requires

### Why base64 plots over cloud storage?
- No S3/GCS setup required
- Simpler deployment
- Plots are ephemeral (no need to store)
- Frontend just renders <img src="data:image/png;base64,...">

### Why OpenAI/Groq over local models?
- Free tier sufficient for portfolio project
- Much faster inference (<1s vs 5-10s)
- Better explanation quality
- Can switch to local models if needed

### Why walk-forward backtest?
- More realistic than simple train/test split
- Shows model performance on rolling windows
- Standard practice in time-series validation

### Why no authentication?
- Out of scope for portfolio project
- Focus on core functionality
- Can add later if deploying for real users

---

**Document Version:** 1.0
**Last Updated:** 2026-02-02
**Maintainer:** Sid Valecha
