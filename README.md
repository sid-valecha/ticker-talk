# Ticker-Talk

Natural-language stock analysis application.

## Overview

Ticker-Talk is an app where:
- Python computes all metrics deterministically
- An LLM is used only for explanation
- Real market data is cached aggressively
- Forecasting is responsible and backtested

## Current State (Scaffold Only)

This repository contains only the base scaffold. The following features are **not yet implemented**:
- Data fetching and caching
- Technical indicators and metrics
- Forecasting models
- LLM integration for explanations
- Visualizations

## Project Structure

```
ticker-talk/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── main.py    # API entry point (/health endpoint)
│   │   ├── config.py  # Environment configuration
│   │   ├── data/      # (empty) Data fetching module
│   │   ├── compute/   # (empty) Metrics module
│   │   ├── plots/     # (empty) Visualization module
│   │   └── llm/       # (empty) LLM integration module
│   └── requirements.txt
├── frontend/          # Next.js frontend
│   └── app/
│       └── page.tsx   # Single page calling /health
└── .env.example
```

## Prerequisites

- Python 3.10+ with conda
- Node.js 18+
- Conda environment `ttalk`

## Running Locally

### Backend

```bash
conda activate ttalk
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs on http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns `{"status": "ok"}` |

## License

MIT
