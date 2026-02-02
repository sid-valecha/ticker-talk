# Ticker-Talk Implementation Plan

## Project Context

**What:** Natural-language stock analysis app with forecasting and explanations
**Current State:** Bare scaffold (FastAPI + Next.js, only /health endpoint)
**Goal:** Build 8 steps incrementally, each fully tested and working

## Core Constraints (LOCKED)

- **Data source:** Alpha Vantage daily adjusted (free tier: 25 calls/day)
- **Cache:** SQLite with 24-hour TTL per ticker
- **Forecast model:** ARIMA (statsmodels) for MVP
- **Forecast horizons:** 7 or 30 days only
- **Backtesting:** Walk-forward only
- **Charts:** matplotlib → base64 PNG (no cloud storage)
- **LLM:** OpenAI/Groq for intent parsing + explanation ONLY (never computes)
- **CSV support:** Step 6 (after core stock flow works)
- **Testing:** Unit tests per step

## Implementation Steps

---

### **STEP 1: Data Ingestion + Caching**
**Goal:** Stock data fetching, SQLite caching, /analyze endpoint returning metadata

#### 1.1 Dependencies
Add to `backend/requirements.txt`:
```
requests
aiosqlite  # or sqlite3 if sync is fine
pandas
numpy
```

#### 1.2 SQLite Cache Schema
**File:** `backend/app/data/cache.py`

**Tables:**
```sql
CREATE TABLE ticker_cache (
    ticker TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,  -- JSON array of OHLCV records
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count INTEGER,
    min_date TEXT,
    max_date TEXT
);

CREATE TABLE request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT,
    ticker TEXT,
    cache_hit BOOLEAN,
    latency_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Functions:**
- `get_cached_data(ticker) -> Optional[dict]` - Returns cached data if < 24h old
- `store_data(ticker, data, min_date, max_date, row_count)` - Stores fetched data
- `log_request(endpoint, ticker, cache_hit, latency_ms)` - Logs metrics
- `is_cache_valid(fetched_at) -> bool` - Checks if timestamp < 24h old

#### 1.3 Alpha Vantage Client
**File:** `backend/app/data/alpha_vantage.py`

**Functions:**
- `fetch_daily_adjusted(ticker: str) -> pd.DataFrame`
  - Call `TIME_SERIES_DAILY_ADJUSTED`
  - Return DataFrame with columns: `date, open, high, low, close, adj_close, volume`
  - Handle rate limits (return None if rate limited)
  - Handle API errors (invalid ticker, network errors)

**Config update:**
Add to `backend/app/config.py`:
```python
ALPHA_VANTAGE_API_KEY: str
ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"
CACHE_TTL_HOURS: int = 24
```

#### 1.4 Demo Mode (Preloaded Tickers)
**File:** `backend/app/data/demo_data.py`

Preload 3 tickers on first run: **AAPL, MSFT, GOOGL**
- On server start, check if these exist in cache
- If not and no API key, fail with helpful error
- If API key exists, pre-fetch and cache them

#### 1.5 /analyze Endpoint (Metadata Only)
**File:** `backend/app/api/analyze.py`

**Request schema:**
```python
class AnalyzeRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None  # YYYY-MM-DD, default: 1 year ago
    end_date: Optional[str] = None    # YYYY-MM-DD, default: today
```

**Response schema (Step 1 only):**
```python
class AnalyzeResponse(BaseModel):
    metadata: Metadata
    # indicators, forecast, plots, explanation added in later steps

class Metadata(BaseModel):
    ticker: str
    cache_hit: bool
    data_last_updated: str  # ISO timestamp
    source: str  # "alpha_vantage" or "demo"
    row_count: int
    min_date: str
    max_date: str
```

**Logic:**
1. Check cache → if hit and valid, return cached data + `cache_hit=True`
2. If miss, fetch from Alpha Vantage
3. If rate limited or API down, try demo data
4. Store in cache
5. Log request metrics
6. Return metadata

**Mount in main.py:**
```python
from app.api import analyze
app.include_router(analyze.router, prefix="/api")
```

#### 1.6 Unit Tests
**File:** `backend/tests/test_cache.py`
- Test cache hit/miss
- Test TTL expiration
- Test demo mode fallback

**File:** `backend/tests/test_alpha_vantage.py`
- Test successful fetch
- Test rate limit handling
- Test invalid ticker

#### 1.7 Verification
```bash
# Start backend
uvicorn app.main:app --reload

# Test
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Expected: metadata with cache_hit, row_count, dates
```

---

### **STEP 2: Technical Indicators**
**Goal:** Add returns, MA, volatility, RSI to /analyze response

#### 2.1 Dependencies
Already have pandas/numpy from Step 1.

#### 2.2 Indicator Functions
**File:** `backend/app/compute/indicators.py`

**Functions:**
```python
def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Daily returns: (price[t] - price[t-1]) / price[t-1]"""
    return df['adj_close'].pct_change()

def compute_moving_average(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average"""
    return df['adj_close'].rolling(window=window).mean()

def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling standard deviation of returns"""
    returns = compute_returns(df)
    return returns.rolling(window=window).std()

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to dataframe"""
    df = df.copy()
    df['returns'] = compute_returns(df)
    df['ma_20'] = compute_moving_average(df, window=20)
    df['volatility_20'] = compute_volatility(df, window=20)
    df['rsi_14'] = compute_rsi(df, window=14)
    return df
```

#### 2.3 Update Response Schema
**File:** `backend/app/models/schemas.py`

```python
class IndicatorSummary(BaseModel):
    latest_return: float
    ma_20_latest: float
    volatility_20_latest: float
    rsi_14_latest: float
    avg_return_30d: float
    avg_volatility_30d: float

class AnalyzeResponse(BaseModel):
    metadata: Metadata
    indicators: IndicatorSummary  # NEW
    # forecast, plots, explanation added later
```

#### 2.4 Update /analyze Endpoint
**File:** `backend/app/api/analyze.py`

After fetching data:
1. Compute indicators using `compute_all_indicators()`
2. Extract summary stats (latest values, 30-day averages)
3. Return in response

#### 2.5 Unit Tests
**File:** `backend/tests/test_indicators.py`
- Test each indicator on known data
- Test edge cases (short series, NaN handling)

#### 2.6 Verification
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Expected: metadata + indicators section
```

---

### **STEP 3: Forecasting + Backtesting**
**Goal:** ARIMA forecast with confidence intervals and walk-forward backtest

#### 3.1 Dependencies
Add to `requirements.txt`:
```
statsmodels
scikit-learn  # for metrics
```

#### 3.2 ARIMA Forecast
**File:** `backend/app/compute/forecast_arima.py`

**Functions:**
```python
def fit_arima(
    series: pd.Series,
    order: tuple = (5, 1, 0)  # default ARIMA(5,1,0)
) -> ARIMAResults:
    """Fit ARIMA model to series"""
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=order)
    return model.fit()

def forecast_arima(
    series: pd.Series,
    horizon: int,  # 7 or 30
    order: tuple = (5, 1, 0)
) -> dict:
    """
    Returns:
    {
        'forecast': [values],  # predicted prices
        'lower_ci': [values],  # 95% lower bound
        'upper_ci': [values],  # 95% upper bound
        'dates': [dates]       # forecast dates
    }
    """
    fitted = fit_arima(series, order)
    forecast_result = fitted.get_forecast(steps=horizon)

    return {
        'forecast': forecast_result.predicted_mean.tolist(),
        'lower_ci': forecast_result.conf_int()['lower adj_close'].tolist(),
        'upper_ci': forecast_result.conf_int()['upper adj_close'].tolist(),
        'dates': pd.date_range(
            series.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        ).strftime('%Y-%m-%d').tolist()
    }
```

#### 3.3 Walk-Forward Backtest
**File:** `backend/app/compute/backtest.py`

**Functions:**
```python
def walk_forward_backtest(
    series: pd.Series,
    horizon: int,
    n_splits: int = 5  # number of validation windows
) -> dict:
    """
    Walk-forward validation:
    - Split data into train/test windows
    - For each split, fit on train, predict on test
    - Compute MAE, RMSE, MAPE

    Returns:
    {
        'predictions': [...],  # all predictions
        'actuals': [...],      # all actual values
        'mae': float,
        'rmse': float,
        'mape': float
    }
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Implementation: rolling window approach
    # Use last n_splits windows of size horizon
    # Fit ARIMA on train, predict horizon steps, compare to actual

    # Return metrics
```

#### 3.4 Update Response Schema
```python
class ForecastResult(BaseModel):
    horizon: int  # 7 or 30
    forecast: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    dates: List[str]

class BacktestMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float
    predictions: List[float]
    actuals: List[float]

class AnalyzeResponse(BaseModel):
    metadata: Metadata
    indicators: IndicatorSummary
    forecast: Optional[ForecastResult] = None  # NEW
    backtest: Optional[BacktestMetrics] = None  # NEW
    # plots, explanation added later
```

#### 3.5 Update Request Schema
```python
class AnalyzeRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_horizon: Optional[int] = None  # NEW: 7 or 30, None = no forecast
```

#### 3.6 Update /analyze Endpoint
If `forecast_horizon` is provided:
1. Run ARIMA forecast
2. Run walk-forward backtest
3. Add to response

#### 3.7 Unit Tests
**File:** `backend/tests/test_forecast.py`
- Test ARIMA on synthetic data
- Test forecast length = horizon

**File:** `backend/tests/test_backtest.py`
- Test walk-forward on known series
- Test metrics calculation

#### 3.8 Verification
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "forecast_horizon": 7}'

# Expected: metadata + indicators + forecast + backtest
```

---

### **STEP 4: Plotting**
**Goal:** Generate base64 PNG charts for price, indicators, and forecast

#### 4.1 Dependencies
Add to `requirements.txt`:
```
matplotlib
```

#### 4.2 Chart Functions
**File:** `backend/app/plots/charts.py`

**Functions:**
```python
def plot_price_and_ma(df: pd.DataFrame) -> str:
    """
    Plot price history with 20-day MA overlay
    Returns: base64 PNG string
    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['adj_close'], label='Price', alpha=0.7)
    ax.plot(df.index, df['ma_20'], label='20-day MA', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_returns_volatility(df: pd.DataFrame) -> str:
    """
    Plot returns and volatility as dual-axis chart
    Returns: base64 PNG string
    """
    # Similar structure, dual y-axis

def plot_forecast(
    historical: pd.Series,
    forecast_data: dict  # from forecast_arima()
) -> str:
    """
    Plot historical prices + forecast with confidence interval
    Returns: base64 PNG string
    """
    # Plot historical as line
    # Plot forecast as line
    # Fill between lower_ci and upper_ci with alpha
```

#### 4.3 Update Response Schema
```python
class Plots(BaseModel):
    price_and_ma: str  # base64 PNG
    returns_volatility: str
    forecast: Optional[str] = None  # only if forecast requested

class AnalyzeResponse(BaseModel):
    metadata: Metadata
    indicators: IndicatorSummary
    forecast: Optional[ForecastResult] = None
    backtest: Optional[BacktestMetrics] = None
    plots: Plots  # NEW
    # explanation added in Step 5
```

#### 4.4 Update /analyze Endpoint
After computing indicators/forecast:
1. Generate all plots
2. Add to response

#### 4.5 Unit Tests
**File:** `backend/tests/test_plots.py`
- Test each plot function returns valid base64
- Test plot generation doesn't crash on edge cases

#### 4.6 Verification
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "forecast_horizon": 7}'

# Expected: plots section with base64 strings
# Decode one to verify it's a valid PNG
```

---

### **STEP 5: LLM Integration**
**Goal:** Intent parsing (optional, for NL queries) + grounded explanation generation

#### 5.1 Dependencies
Add to `requirements.txt`:
```
openai
```

#### 5.2 Config Update
Add to `config.py`:
```python
OPENAI_API_KEY: str = ""
GROQ_API_KEY: str = ""
LLM_PROVIDER: str = "openai"  # or "groq"
LLM_MODEL: str = "gpt-4o-mini"  # or groq model
```

#### 5.3 LLM Client
**File:** `backend/app/llm/client.py`

**Functions:**
```python
def get_llm_client():
    """Return OpenAI or Groq client based on config"""
    if settings.LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(api_key=settings.OPENAI_API_KEY)
    elif settings.LLM_PROVIDER == "groq":
        from openai import OpenAI  # Groq uses OpenAI SDK
        return OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

def call_llm(messages: list, max_tokens: int = 500) -> str:
    """Call LLM with messages, return response text"""
    client = get_llm_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3
    )
    return response.choices[0].message.content
```

#### 5.4 Intent Parsing (Optional Enhancement)
**File:** `backend/app/llm/intent.py`

**Function:**
```python
def parse_intent(query: str) -> dict:
    """
    Parse natural language query into structured request
    Example: "forecast AAPL for 30 days" → {"ticker": "AAPL", "forecast_horizon": 30}

    Returns: dict with ticker, forecast_horizon, start_date, end_date
    """
    system_prompt = """
    Extract stock analysis parameters from user query.
    Return JSON with: ticker, forecast_horizon (7 or 30 or null), start_date, end_date.
    Only valid horizons are 7 or 30 days.
    If no horizon mentioned, set to null.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = call_llm(messages, max_tokens=100)
    return json.loads(response)
```

**Note:** This is optional for MVP. Can start with structured requests only.

#### 5.5 Explanation Generation
**File:** `backend/app/llm/explain.py`

**Function:**
```python
def generate_explanation(analysis_summary: dict) -> str:
    """
    Generate grounded explanation from computed results

    analysis_summary contains:
    - ticker
    - latest indicators (returns, MA, RSI, volatility)
    - forecast if available (predicted trend)
    - backtest metrics if available
    """
    system_prompt = """
    You are a financial analysis assistant.

    STRICT RULES:
    - Base explanation ONLY on provided computed metrics
    - Never give trading advice or recommendations
    - Acknowledge uncertainty and limitations
    - Explain what the numbers mean, not what to do
    - If forecast exists, explain it's a model prediction with uncertainty
    """

    user_prompt = f"""
    Summarize this stock analysis in 2-3 paragraphs:

    {json.dumps(analysis_summary, indent=2)}

    Focus on:
    1. Recent price behavior (returns, volatility, trend)
    2. Technical indicators (MA, RSI)
    3. If forecast: explain prediction and confidence interval
    4. If backtest: explain model accuracy (MAE, RMSE)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return call_llm(messages, max_tokens=500)
```

#### 5.6 Update Response Schema
```python
class AnalyzeResponse(BaseModel):
    metadata: Metadata
    indicators: IndicatorSummary
    forecast: Optional[ForecastResult] = None
    backtest: Optional[BacktestMetrics] = None
    plots: Plots
    explanation: str  # NEW
```

#### 5.7 Update /analyze Endpoint
After computing everything:
1. Build analysis_summary dict from results
2. Call `generate_explanation()`
3. Add to response

#### 5.8 Guardrails
Add validation to refuse:
- Invalid forecast horizons (not 7 or 30)
- Requests for trading advice (detect in intent parsing)
- Intraday data requests

#### 5.9 Unit Tests
**File:** `backend/tests/test_llm.py`
- Test explanation generation with mock data
- Test guardrails (invalid horizons)

#### 5.10 Verification
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "forecast_horizon": 7}'

# Expected: full response with explanation
```

---

### **STEP 6: CSV Upload + Frontend UI**
**Goal:** CSV upload endpoint + complete frontend with forms and charts

#### 6.1 CSV Upload Endpoint
**File:** `backend/app/api/analyze_csv.py`

**Dependencies:**
```
python-multipart  # for file uploads
```

**Endpoint:**
```python
@router.post("/analyze_csv")
async def analyze_csv(
    file: UploadFile,
    forecast_horizon: Optional[int] = None
) -> AnalyzeResponse:
    """
    Analyze uploaded CSV dataset

    Required columns: date, price (or close/adj_close)
    Optional columns: volume, open, high, low
    """
    # Read CSV
    # Validate schema (has date + price columns)
    # Normalize to standard format
    # Store in cache with hash-based key
    # Run same pipeline as stock analysis
    # Return same response format
```

**File:** `backend/app/data/csv_loader.py`

**Functions:**
```python
def validate_csv(df: pd.DataFrame) -> bool:
    """Check required columns exist"""
    required = ['date']
    price_cols = ['price', 'close', 'adj_close']

    has_date = 'date' in df.columns
    has_price = any(col in df.columns for col in price_cols)

    return has_date and has_price

def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CSV to standard format:
    - Parse date column to datetime index
    - Rename price column to adj_close
    - Fill missing OHLCV columns with price if needed
    """
    # Implementation
```

#### 6.2 Frontend Components

**File:** `frontend/app/lib/api.ts`
```typescript
export async function analyzeStock(request: {
  ticker: string;
  forecast_horizon?: number;
  start_date?: string;
  end_date?: string;
}) {
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return response.json();
}

export async function analyzeCSV(file: File, forecast_horizon?: number) {
  const formData = new FormData();
  formData.append('file', file);
  if (forecast_horizon) {
    formData.append('forecast_horizon', String(forecast_horizon));
  }

  const response = await fetch('http://localhost:8000/api/analyze_csv', {
    method: 'POST',
    body: formData,
  });
  return response.json();
}
```

**File:** `frontend/app/components/QueryForm.tsx`
```typescript
// Form with:
// - Ticker input
// - OR file upload
// - Forecast toggle (7 / 30 / none)
// - Submit button
// - Loading state
```

**File:** `frontend/app/components/ChartsPanel.tsx`
```typescript
// Display base64 images:
// - Price + MA chart
// - Returns/Volatility chart
// - Forecast chart (if exists)
```

**File:** `frontend/app/components/ExplanationPanel.tsx`
```typescript
// Display:
// - LLM explanation
// - Metadata (cache hit, data freshness)
// - Disclaimer text
```

**File:** `frontend/app/page.tsx`
```typescript
// Main page:
// - QueryForm
// - On submit: call API, show loading
// - On success: show ChartsPanel + ExplanationPanel
// - On error: show error message
```

#### 6.3 Styling
Use Tailwind (already installed) for basic styling. Keep it minimal.

#### 6.4 Verification
```bash
# Start both servers
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev

# Test in browser:
# - Enter AAPL ticker, request 7-day forecast
# - Upload CSV with sample data
# - Verify charts render and explanation appears
```

---

### **STEP 7: Metrics + Deployment**
**Goal:** Request logging, cache hit metrics, deploy to production

#### 7.1 Metrics Dashboard Endpoint
**File:** `backend/app/api/metrics.py`

**Endpoint:**
```python
@router.get("/metrics")
def get_metrics():
    """
    Return metrics from SQLite:
    - total_requests
    - cache_hit_rate
    - avg_latency_ms
    - requests_by_ticker (top 10)
    """
    # Query request_metrics table
    # Return summary stats
```

#### 7.2 Deployment

**Backend to Railway:**
1. Create `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
```

2. Add Procfile (alternative):
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

3. Set environment variables in Railway dashboard:
   - `ALPHA_VANTAGE_API_KEY`
   - `OPENAI_API_KEY` or `GROQ_API_KEY`
   - `LLM_PROVIDER`

**Frontend to Vercel:**
1. Connect GitHub repo
2. Set build settings:
   - Framework: Next.js
   - Root directory: `frontend`
3. Set environment variable:
   - `NEXT_PUBLIC_API_URL` = Railway backend URL

#### 7.3 Health Endpoint Enhancement
Update `/health` to include:
- Database connection status
- Cache size
- API key presence (not value)

#### 7.4 Verification
```bash
# Test deployed backend
curl https://your-app.railway.app/health

# Test deployed frontend
# Visit https://your-app.vercel.app
# Submit query, verify end-to-end flow
```

---

### **STEP 8: Documentation + Metrics Snapshot**
**Goal:** Complete README, architecture docs, record metrics

#### 8.1 README Update
**File:** `README.md`

Sections:
- Overview (what this does)
- Architecture diagram (text-based or mermaid)
- Features
- API endpoints
- Local setup
- Deployment
- Limitations & disclaimers
- Metrics snapshot
- License

**Disclaimers to include:**
- Not financial advice
- Models have limitations
- Past performance != future results
- For educational purposes only

#### 8.2 Architecture Diagram
Create simple text diagram or mermaid flowchart:
```
User → Frontend → API → Cache Check → Alpha Vantage (if miss)
                    ↓
                  Compute Indicators
                    ↓
                  ARIMA Forecast
                    ↓
                  Generate Plots
                    ↓
                  LLM Explanation
                    ↓
                  Return Response
```

#### 8.3 Metrics Snapshot
After 1-2 weeks of usage, capture:
- Total queries processed
- Cache hit rate %
- Average latency (ms)
- Most queried tickers
- CSV uploads count

Add to README as "Performance Metrics" section.

#### 8.4 Demo Instructions
Add to README:
- How to run demo mode (preloaded tickers)
- Example queries to try
- Expected output screenshots (optional)

---

## Critical Files to Create/Modify

### Backend
- `app/data/cache.py` - SQLite cache layer
- `app/data/alpha_vantage.py` - API client
- `app/data/csv_loader.py` - CSV validation
- `app/data/demo_data.py` - Preloaded tickers
- `app/compute/indicators.py` - Technical indicators
- `app/compute/forecast_arima.py` - ARIMA forecasting
- `app/compute/backtest.py` - Walk-forward validation
- `app/plots/charts.py` - matplotlib plotting
- `app/llm/client.py` - LLM client abstraction
- `app/llm/explain.py` - Explanation generation
- `app/api/analyze.py` - Stock analysis endpoint
- `app/api/analyze_csv.py` - CSV analysis endpoint
- `app/api/metrics.py` - Metrics endpoint
- `app/models/schemas.py` - Pydantic models
- `app/config.py` - Environment config
- `requirements.txt` - Python dependencies

### Frontend
- `app/lib/api.ts` - API client
- `app/components/QueryForm.tsx` - Input form
- `app/components/ChartsPanel.tsx` - Chart display
- `app/components/ExplanationPanel.tsx` - Explanation display
- `app/page.tsx` - Main page
- `next.config.ts` - Config for CORS/API URL

### Deployment
- `railway.toml` or `Procfile` - Railway config
- `.env.example` - Environment template
- `docker-compose.yml` (optional) - Local dev

### Documentation
- `README.md` - Complete guide
- `ARCHITECTURE.md` - Design decisions
- `CLAUDE.md` - Implementation notes for handoff

---

## Testing Strategy

Each step includes unit tests:
- **Step 1:** Cache hit/miss, Alpha Vantage fetch, demo mode
- **Step 2:** Indicator calculations (deterministic)
- **Step 3:** ARIMA fitting, backtest metrics
- **Step 4:** Plot generation (valid base64)
- **Step 5:** LLM explanation (mock responses), guardrails
- **Step 6:** CSV validation, normalization

Run tests after each step:
```bash
cd backend
pytest tests/
```

---

## Success Criteria (Done = Shipped)

- [ ] AAPL and MSFT work end-to-end (cache → indicators → forecast → plots → explanation)
- [ ] One CSV upload works (valid data → analysis)
- [ ] 7-day and 30-day forecasts both work
- [ ] Walk-forward backtest returns MAE/RMSE
- [ ] Charts generated and displayed in frontend
- [ ] LLM explanation grounded in computed metrics
- [ ] Cache reduces API calls (hit rate > 50% after warmup)
- [ ] Deployed to Railway + Vercel
- [ ] README + ARCHITECTURE.md complete
- [ ] Metrics logged and queryable via /metrics

---

## Notes for Handoff to Codex

**Current state:** Scaffold only (FastAPI + Next.js, /health endpoint)

**Key decisions locked:**
- SQLite cache with 24h TTL
- ARIMA for forecasting (not Prophet for MVP)
- matplotlib → base64 PNG (no cloud storage)
- OpenAI/Groq for LLM (not custom orchestration layer)
- CSV upload in Step 6 (not Step 1)
- Tests per step (not at end)

**Implementation order:** Steps 1-8 sequentially. Each step must work before moving to next.

**No scope creep:**
- No auth/user accounts
- No portfolio tracking
- No real-time data
- No trading signals
- No complex UI animations

**Focus:** Build simple, working, testable code. Ship fast.
