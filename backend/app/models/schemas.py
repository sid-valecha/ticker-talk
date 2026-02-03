from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_horizon: Optional[int] = Field(None, ge=7, le=30)  # 7 or 30 days, None = no forecast

    @field_validator("forecast_horizon")
    @classmethod
    def validate_horizon(cls, v):
        if v is not None and v not in [7, 30]:
            raise ValueError("forecast_horizon must be 7 or 30 days")
        return v


class Metadata(BaseModel):
    ticker: str
    cache_hit: bool
    data_last_updated: str
    source: str
    row_count: int
    min_date: str
    max_date: str


class IndicatorSummary(BaseModel):
    latest_price: float
    latest_return: float
    ma_20_latest: float
    volatility_20_latest: float
    rsi_14_latest: float
    avg_return_30d: float
    avg_volatility_30d: float


class ForecastResult(BaseModel):
    horizon: int
    forecast: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    dates: List[str]
    trend: str  # "upward", "downward", or "flat"


class BacktestMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float


class Plots(BaseModel):
    price_and_ma: str  # base64 PNG
    returns_volatility: str  # base64 PNG
    rsi: str  # base64 PNG
    forecast: Optional[str] = None  # base64 PNG, only if forecast requested


class AnalyzeResponse(BaseModel):
    metadata: Metadata
    indicators: Optional[IndicatorSummary] = None
    forecast: Optional[ForecastResult] = None
    backtest: Optional[BacktestMetrics] = None
    plots: Optional[Plots] = None
