import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import analyze, metrics
from app.config import settings
from app.data.cache import init_db
from app.data.demo_data import DEMO_TICKERS, preload_demo_data_if_needed

logger = logging.getLogger(__name__)

# Rate limiter: 10 requests/minute per IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Ticker-Talk API",
    description="Natural-language stock analysis API",
    version="0.1.0",
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return generic error.

    Prevents stack traces from leaking to the client.
    """
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.on_event("startup")
def on_startup() -> None:
    # Validate configuration
    if not settings.ALPHA_VANTAGE_API_KEY:
        logger.warning(
            "ALPHA_VANTAGE_API_KEY not set. API fetching disabled, demo data only."
        )

    init_db()
    preload_demo_data_if_needed()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@app.get("/api/tickers")
def get_available_tickers():
    """Get list of available demo tickers."""
    return {"tickers": DEMO_TICKERS}


app.include_router(analyze.router, prefix="/api")
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
