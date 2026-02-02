from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    ENVIRONMENT: str = "development"
    ALPHA_VANTAGE_API_KEY: str = ""
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"
    CACHE_TTL_HOURS: int = 24

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
