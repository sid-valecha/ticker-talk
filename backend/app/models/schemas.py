from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class Metadata(BaseModel):
    ticker: str
    cache_hit: bool
    data_last_updated: str
    source: str
    row_count: int
    min_date: str
    max_date: str


class AnalyzeResponse(BaseModel):
    metadata: Metadata
