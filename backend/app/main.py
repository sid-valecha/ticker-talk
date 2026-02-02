from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyze
from app.config import settings
from app.data.cache import init_db
from app.data.demo_data import preload_demo_data_if_needed

app = FastAPI(
    title="Ticker-Talk API",
    description="Natural-language stock analysis API",
    version="0.1.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    preload_demo_data_if_needed()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": settings.ENVIRONMENT}


app.include_router(analyze.router, prefix="/api")
