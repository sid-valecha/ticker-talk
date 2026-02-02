# Claude Code Implementation Notes

**Purpose:** This document captures insights, patterns, and gotchas discovered during planning and implementation. Use this as a reference when coding or debugging.

**Audience:** Future developers (human or AI) continuing this project

**Last Updated:** 2026-02-02 (Planning Phase)

---

## Project Context

**Goal:** Build a stock analysis web app with natural language interface
**Tech Stack:** FastAPI + Next.js + SQLite + OpenAI/Groq
**Constraints:** 25 API calls/day (Alpha Vantage free tier), no trading advice

**Starting Point:** Bare scaffold with FastAPI /health endpoint and Next.js hello-world page

**Implementation Strategy:** 8 sequential steps, each fully tested before moving on

---

## Key Implementation Patterns

### 1. Cache-First Architecture

**Pattern:** Always check cache before external API calls

```python
# Good pattern (used throughout project)
def get_stock_data(ticker: str) -> pd.DataFrame:
    # 1. Check cache
    cached = cache.get(ticker)
    if cached and cache.is_valid(cached):
        return cached['data']

    # 2. Fetch if needed
    data = alpha_vantage.fetch(ticker)

    # 3. Store in cache
    cache.store(ticker, data)

    return data
```

**Why:** Alpha Vantage limits to 25 calls/day. Without caching, you'll hit limits during development.

**Gotcha:** Remember to check TTL (24 hours). Don't cache indefinitely.

### 2. Deterministic Computation

**Pattern:** All numerical computations happen in pandas/numpy, never in LLM

```python
# Good: Deterministic calculation
def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df['adj_close'].pct_change()

# Bad: LLM computing numbers (NEVER DO THIS)
def compute_returns_llm(df: pd.DataFrame) -> pd.Series:
    prompt = f"Calculate returns for {df['adj_close'].tolist()}"
    return llm.call(prompt)  # ❌ Non-deterministic, wrong!
```

**Why:** Financial calculations must be precise and reproducible. LLMs are for explanation only.

**Gotcha:** Don't be tempted to use LLMs for "smart" indicator selection or parameter tuning. Hard-code or make user-configurable.

### 3. LLM Grounding

**Pattern:** Only pass computed results to LLM, never raw data

```python
# Good: LLM sees summary stats only
summary = {
    "ticker": "AAPL",
    "latest_price": 174.85,
    "returns_30d_avg": 0.0012,
    "volatility_30d": 0.018,
    "rsi": 58,
    "forecast_trend": "upward",
    "forecast_mae": 2.15
}
explanation = llm.explain(summary)

# Bad: Sending raw dataframe (too much, not structured)
explanation = llm.explain(df.to_json())  # ❌ Wastes tokens, less focused
```

**Why:** LLMs work better with structured summaries. Reduces hallucination by limiting scope.

**Gotcha:** If explanation seems ungrounded, check that summary includes all necessary context.

### 4. Pydantic Validation Everywhere

**Pattern:** Define strict schemas for API requests and responses

```python
class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., pattern=r'^[A-Z]{1,5}$')
    forecast_horizon: Optional[int] = Field(None, ge=7, le=30)

    @validator('forecast_horizon')
    def validate_horizon(cls, v):
        if v is not None and v not in [7, 30]:
            raise ValueError('horizon must be 7 or 30')
        return v
```

**Why:** Fail fast with clear errors. Frontend gets actionable validation messages.

**Gotcha:** Don't forget to handle optional fields (forecast can be None).

### 5. Metrics Logging from Day 1

**Pattern:** Log every request to SQLite, even in Step 1

```python
def log_request(endpoint: str, ticker: str, cache_hit: bool, latency_ms: int):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO request_metrics (endpoint, ticker, cache_hit, latency_ms)
        VALUES (?, ?, ?, ?)
    """, (endpoint, ticker, cache_hit, latency_ms))
    conn.commit()
```

**Why:** Metrics tell you cache hit rate, popular tickers, performance. Needed for README metrics section.

**Gotcha:** Wrap in try/except so metrics failure doesn't break requests.

---

## Module-Specific Notes

### Alpha Vantage Client (`app/data/alpha_vantage.py`)

**API Endpoint:**
```
https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&apikey=YOUR_KEY
```

**Response Structure:**
```json
{
    "Meta Data": {...},
    "Time Series (Daily)": {
        "2026-02-01": {
            "1. open": "174.00",
            "2. high": "175.50",
            "3. low": "173.80",
            "4. close": "174.85",
            "5. adjusted close": "174.85",
            "6. volume": "52000000",
            ...
        }
    }
}
```

**Parsing:**
```python
def parse_response(json_data: dict) -> pd.DataFrame:
    time_series = json_data.get("Time Series (Daily)", {})

    records = []
    for date_str, values in time_series.items():
        records.append({
            'date': pd.to_datetime(date_str),
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'adj_close': float(values['5. adjusted close']),
            'volume': int(values['6. volume'])
        })

    df = pd.DataFrame(records)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)  # Ascending date order
    return df
```

**Rate Limiting:**
- Free tier: 25 requests/day, 5 requests/minute
- Check for error message: `"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute..."`
- Return None and log if rate limited

**Error Cases:**
- Invalid ticker: `"Error Message": "Invalid API call..."`
- Network error: Handle requests.RequestException
- API key missing: Check on server start, fail fast

### SQLite Cache (`app/data/cache.py`)

**File Location:** `backend/data_cache/ticker_talk.db`

**Initialization:**
```python
def init_db():
    conn = sqlite3.connect('data_cache/ticker_talk.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_cache (
            ticker TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER,
            min_date TEXT,
            max_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT,
            ticker TEXT,
            cache_hit BOOLEAN,
            latency_ms INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
```

**TTL Check:**
```python
def is_cache_valid(fetched_at_str: str, ttl_hours: int = 24) -> bool:
    fetched_at = datetime.fromisoformat(fetched_at_str)
    age = datetime.now() - fetched_at
    return age.total_seconds() < (ttl_hours * 3600)
```

**Gotcha:** SQLite datetime columns are stored as strings. Use `datetime.fromisoformat()` to parse.

### Technical Indicators (`app/compute/indicators.py`)

**Returns Calculation:**
```python
def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Daily returns: (price[t] - price[t-1]) / price[t-1]"""
    return df['adj_close'].pct_change()
```

**Moving Average:**
```python
def compute_moving_average(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average (SMA)"""
    return df['adj_close'].rolling(window=window).mean()
```

**Volatility:**
```python
def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized volatility from rolling std of returns"""
    returns = compute_returns(df)
    # Daily std * sqrt(252 trading days) = annualized volatility
    return returns.rolling(window=window).std() * np.sqrt(252)
```

**RSI (Relative Strength Index):**
```python
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over window
    """
    delta = df['adj_close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

**Gotcha:** First `window` rows will be NaN. Handle with `.dropna()` or `.fillna(0)` depending on use case.

### ARIMA Forecasting (`app/compute/forecast_arima.py`)

**Model Selection:**
- Default order: (5, 1, 0) - 5 autoregressive terms, 1st-order differencing, no moving average
- Works well for 7-30 day horizons
- Can make configurable later if needed

**Fitting:**
```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(series: pd.Series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    return model.fit()
```

**Forecasting:**
```python
def forecast_arima(series: pd.Series, horizon: int, order=(5, 1, 0)):
    fitted = fit_arima(series, order)
    forecast_result = fitted.get_forecast(steps=horizon)

    return {
        'forecast': forecast_result.predicted_mean.tolist(),
        'lower_ci': forecast_result.conf_int()['lower adj_close'].tolist(),
        'upper_ci': forecast_result.conf_int()['upper adj_close'].tolist(),
        'dates': pd.date_range(
            series.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'  # Daily frequency
        ).strftime('%Y-%m-%d').tolist()
    }
```

**Gotcha:** ARIMA requires at least ~60 data points for reliable fitting. Check series length and warn user if too short.

**Confidence Interval:** Default is 95% (alpha=0.05). Adjustable via `get_forecast(steps, alpha=0.05)`.

### Walk-Forward Backtest (`app/compute/backtest.py`)

**Concept:** Simulate real-world usage where you train on past data and predict future

**Implementation:**
```python
def walk_forward_backtest(series: pd.Series, horizon: int, n_splits: int = 5):
    """
    Split data into n_splits windows:
    - Each window: train on all data up to t, predict next horizon steps
    - Move forward by horizon steps, repeat

    Returns metrics: MAE, RMSE, MAPE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    predictions = []
    actuals = []

    # Need at least 60 points for train + horizon for test
    min_train_size = 60
    total_length = len(series)

    # Calculate split points
    for i in range(n_splits):
        test_end = total_length - (n_splits - i - 1) * horizon
        test_start = test_end - horizon
        train_end = test_start

        if train_end < min_train_size:
            continue  # Not enough training data

        train_data = series[:train_end]
        test_data = series[test_start:test_end]

        # Fit and predict
        fitted = fit_arima(train_data)
        forecast = fitted.get_forecast(steps=horizon)
        pred = forecast.predicted_mean.values

        predictions.extend(pred)
        actuals.extend(test_data.values)

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'predictions': predictions,
        'actuals': actuals
    }
```

**Gotcha:** MAPE (Mean Absolute Percentage Error) can be infinite if actual price is 0. Add check: `if actual == 0: continue`.

### Plotting (`app/plots/charts.py`)

**Base Pattern:**
```python
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
```

**Price + MA Chart:**
```python
def plot_price_and_ma(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df.index, df['adj_close'], label='Price', linewidth=2)
    ax.plot(df.index, df['ma_20'], label='20-day MA', linewidth=2, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price History with Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return plot_to_base64(fig)
```

**Forecast Chart:**
```python
def plot_forecast(historical: pd.Series, forecast_data: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Historical prices
    ax.plot(historical.index, historical.values, label='Historical', linewidth=2)

    # Forecast
    forecast_dates = pd.to_datetime(forecast_data['dates'])
    ax.plot(forecast_dates, forecast_data['forecast'],
            label='Forecast', linewidth=2, linestyle='--', color='orange')

    # Confidence interval
    ax.fill_between(
        forecast_dates,
        forecast_data['lower_ci'],
        forecast_data['upper_ci'],
        alpha=0.2,
        color='orange',
        label='95% Confidence'
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return plot_to_base64(fig)
```

**Gotcha:** Always call `plt.close(fig)` after saving to avoid memory leaks. matplotlib figures persist in memory otherwise.

### LLM Client (`app/llm/client.py`)

**OpenAI Integration:**
```python
from openai import OpenAI

def get_openai_client():
    return OpenAI(api_key=settings.OPENAI_API_KEY)

def call_openai(messages: list, max_tokens: int = 500):
    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,  # "gpt-4o-mini"
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3  # Lower = more deterministic
    )
    return response.choices[0].message.content
```

**Groq Integration:**
```python
from openai import OpenAI  # Groq uses same SDK!

def get_groq_client():
    return OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

def call_groq(messages: list, max_tokens: int = 500):
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or other Groq model
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3
    )
    return response.choices[0].message.content
```

**Provider Abstraction:**
```python
def call_llm(messages: list, max_tokens: int = 500) -> str:
    if settings.LLM_PROVIDER == "openai":
        return call_openai(messages, max_tokens)
    elif settings.LLM_PROVIDER == "groq":
        return call_groq(messages, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
```

**Gotcha:** Groq uses OpenAI SDK but different base_url. Both work with same code pattern.

### LLM Explanation (`app/llm/explain.py`)

**System Prompt (Critical - Enforces Guardrails):**
```python
SYSTEM_PROMPT = """
You are a financial analysis assistant that explains stock analysis results.

STRICT RULES:
1. Base all explanations ONLY on the provided computed metrics
2. Never give trading advice, buy/sell recommendations, or price predictions as facts
3. Always acknowledge uncertainty and model limitations
4. Explain what the numbers mean and what the model predicts, not what users should do
5. If discussing forecasts, emphasize they are statistical predictions with confidence intervals
6. Cite specific metrics from the input (e.g., "The RSI of 58 suggests...")

FORBIDDEN PHRASES:
- "You should buy/sell"
- "This is a good/bad investment"
- "The stock will definitely..."
- "This indicates you should..."

REQUIRED DISCLAIMERS:
- Mention model limitations (e.g., "past performance does not guarantee future results")
- Note uncertainty (e.g., "the forecast has a mean absolute error of X, meaning...")
"""
```

**User Prompt Template:**
```python
def build_user_prompt(analysis_summary: dict) -> str:
    return f"""
Summarize this stock analysis in 2-3 paragraphs for a user:

Ticker: {analysis_summary['ticker']}
Latest Price: ${analysis_summary['latest_price']}

Technical Indicators (latest values):
- Daily Return: {analysis_summary['latest_return']:.2%}
- 20-day Moving Average: ${analysis_summary['ma_20_latest']:.2f}
- Volatility (20-day): {analysis_summary['volatility_20_latest']:.2%}
- RSI (14-day): {analysis_summary['rsi_14_latest']:.1f}

30-day Averages:
- Avg Daily Return: {analysis_summary['avg_return_30d']:.2%}
- Avg Volatility: {analysis_summary['avg_volatility_30d']:.2%}

{f'''
Forecast ({analysis_summary['forecast_horizon']}-day):
- Predicted Trend: {analysis_summary['forecast_trend']}
- Confidence Interval: ${analysis_summary['forecast_lower']:.2f} - ${analysis_summary['forecast_upper']:.2f}

Backtest Metrics:
- Mean Absolute Error: ${analysis_summary['backtest_mae']:.2f}
- RMSE: ${analysis_summary['backtest_rmse']:.2f}
- MAPE: {analysis_summary['backtest_mape']:.1f}%
''' if 'forecast_horizon' in analysis_summary else ''}

Focus on:
1. Recent price behavior and what indicators suggest about current momentum
2. Volatility and risk level
3. If forecast exists: what the model predicts and its accuracy based on backtest
4. Limitations and uncertainty
"""
```

**Gotcha:** If forecast is not requested, summary won't have forecast fields. Use conditional formatting.

---

## Common Gotchas

### 1. Pandas Date Index
**Problem:** ARIMA expects sorted ascending dates, but some operations return descending

**Solution:**
```python
df.sort_index(inplace=True)  # Always sort after creating DataFrame
series = df['adj_close'].sort_index()  # Ensure series is sorted too
```

### 2. NaN Handling in Indicators
**Problem:** First `window` rows are NaN after rolling calculations

**Solution:**
```python
# Option 1: Drop NaNs before plotting
df_clean = df.dropna()

# Option 2: Forward-fill (use cautiously)
df['ma_20'].fillna(method='ffill', inplace=True)

# Option 3: Return latest non-NaN values for summary
latest_ma = df['ma_20'].dropna().iloc[-1]
```

### 3. SQLite Connection Management
**Problem:** SQLite doesn't handle concurrent writes well

**Solution:**
```python
# Use context manager for connections
def get_db_connection():
    return sqlite3.connect('data_cache/ticker_talk.db')

# Always close connections
with get_db_connection() as conn:
    conn.execute(...)
    conn.commit()
# Connection auto-closed
```

**Gotcha:** If deploying with multiple workers, use `check_same_thread=False` or switch to Postgres.

### 4. Alpha Vantage Date Format
**Problem:** API returns dates as "2026-02-01" but different APIs use different formats

**Solution:**
```python
# Always parse explicitly
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
```

### 5. Base64 Image Encoding
**Problem:** Frontend needs `data:image/png;base64,` prefix for inline images

**Solution:**
```python
# Backend returns just base64 string
base64_str = plot_to_base64(fig)

# Frontend adds prefix
<img src={`data:image/png;base64,${base64_str}`} />
```

### 6. CORS Configuration
**Problem:** Frontend can't call backend API due to CORS

**Solution (backend):**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7. Environment Variable Loading
**Problem:** pydantic-settings doesn't auto-load .env in production

**Solution:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ALPHA_VANTAGE_API_KEY: str

    class Config:
        env_file = ".env"  # Load from .env in development
        env_file_encoding = "utf-8"
        # In production, Railway/Vercel inject env vars directly
```

**Gotcha:** Always set `env_file` even for production. It won't break if file doesn't exist.

---

## Testing Patterns

### Unit Test Structure
```python
# backend/tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from app.compute.indicators import compute_returns, compute_rsi

def test_compute_returns():
    # Create synthetic data
    data = {'adj_close': [100, 102, 101, 105, 103]}
    df = pd.DataFrame(data)

    returns = compute_returns(df)

    # First value is NaN (no previous price)
    assert pd.isna(returns.iloc[0])

    # Second value: (102 - 100) / 100 = 0.02
    assert np.isclose(returns.iloc[1], 0.02, atol=1e-6)

def test_rsi_range():
    # RSI should always be 0-100
    data = {'adj_close': np.random.rand(100) * 100}
    df = pd.DataFrame(data)

    rsi = compute_rsi(df, window=14)
    rsi_clean = rsi.dropna()

    assert (rsi_clean >= 0).all()
    assert (rsi_clean <= 100).all()
```

### Integration Test Pattern
```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_endpoint():
    response = client.post(
        "/api/analyze",
        json={"ticker": "AAPL", "forecast_horizon": 7}
    )

    assert response.status_code == 200
    data = response.json()

    assert "metadata" in data
    assert "indicators" in data
    assert "forecast" in data
    assert data["metadata"]["ticker"] == "AAPL"
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All unit tests pass (`pytest backend/tests/`)
- [ ] AAPL and MSFT queries work end-to-end locally
- [ ] CSV upload works with sample data
- [ ] Cache hit rate > 50% after warmup (query same ticker twice)
- [ ] Forecast generates for 7 and 30 days
- [ ] Backtest returns valid MAE/RMSE
- [ ] LLM explanation appears and is grounded (no hallucinations)
- [ ] Charts render correctly in frontend

### Railway Backend
- [ ] Create new project
- [ ] Connect GitHub repo
- [ ] Set environment variables:
  - `ALPHA_VANTAGE_API_KEY`
  - `OPENAI_API_KEY` or `GROQ_API_KEY`
  - `LLM_PROVIDER=openai` or `groq`
  - `LLM_MODEL=gpt-4o-mini` or groq model
- [ ] Add `railway.toml` or `Procfile`
- [ ] Deploy and check logs
- [ ] Test `/health` endpoint

### Vercel Frontend
- [ ] Connect GitHub repo
- [ ] Set root directory to `frontend`
- [ ] Set environment variable: `NEXT_PUBLIC_API_URL=<railway-url>`
- [ ] Deploy
- [ ] Test UI end-to-end

### Post-Deployment
- [ ] Submit 5-10 test queries to warm cache
- [ ] Check `/metrics` endpoint for cache hit rate
- [ ] Monitor Alpha Vantage API usage (should be < 25/day)
- [ ] Capture metrics snapshot for README
- [ ] Add deployed URLs to README

---

## README Metrics Section Template

```markdown
## Performance Metrics

**Snapshot Date:** 2026-02-10 (1 week after launch)

### Usage
- **Total Queries:** 487
- **Unique Tickers:** 23
- **CSV Uploads:** 12
- **Forecast Requests:** 341 (70% of queries)

### Performance
- **Cache Hit Rate:** 73%
- **Avg Latency (cached):** 1.2s
- **Avg Latency (uncached):** 3.5s
- **Alpha Vantage API Calls:** 18/25 daily limit (28% utilization)

### Top Tickers
1. AAPL (143 queries)
2. MSFT (89 queries)
3. GOOGL (67 queries)
4. TSLA (42 queries)
5. NVDA (38 queries)

### Forecast Accuracy
- **Mean Absolute Error (MAE):** $2.15 avg across all forecasts
- **RMSE:** $3.42
- **MAPE:** 1.8%

*Note: Accuracy varies by ticker volatility and horizon. 7-day forecasts more accurate than 30-day.*
```

---

## Future Work Ideas

### Short-term (Easy Adds)
- [ ] Model comparison: ARIMA vs Prophet side-by-side
- [ ] More indicators: Bollinger Bands, MACD, Stochastic
- [ ] Custom date range picker in UI (currently defaults to 1 year)
- [ ] Export analysis to JSON for external use

### Medium-term (More Work)
- [ ] Multiple ticker comparison (overlay 2-3 stocks on one chart)
- [ ] Sector analysis (group tickers by sector)
- [ ] Email alerts for price thresholds (requires user accounts)
- [ ] Shareable analysis links (generate unique URLs)

### Long-term (Significant Changes)
- [ ] Real-time data (websocket updates)
- [ ] Portfolio optimization (modern portfolio theory)
- [ ] Sentiment analysis from news/Twitter
- [ ] Multi-step LLM agent for complex queries
- [ ] Fine-tuned model for financial explanations

---

**Document Maintained By:** Claude Code (via Anthropic)
**Created:** 2026-02-02 (Planning Phase)
**Next Update:** After Step 4 completion (Plotting)
