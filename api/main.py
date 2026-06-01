"""
Phase 6 — FastAPI Application for Speech Emotion Recognition.
Modularised version.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from api.dependencies import load_predictor, unload_predictor
from api.routes import system, predict, contact_center, psychology

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up on shutdown."""
    load_predictor()
    yield  # Application runs here
    unload_predictor()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Speech Emotion Recognition API",
    description=(
        "REST API for Speech Emotion Recognition (SER) — "
        "supports contact center agent routing and psychologist pre-assessment reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ───────────────────────────────────────────────────────────

app.include_router(system.router)
app.include_router(predict.router)
app.include_router(contact_center.router)
app.include_router(psychology.router)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
