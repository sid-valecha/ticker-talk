import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import analyze
from app.config import settings
from app.data.cache import init_db
from app.data.demo_data import preload_demo_data_if_needed

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    return {"status": "ok"}


app.include_router(analyze.router, prefix="/api")
