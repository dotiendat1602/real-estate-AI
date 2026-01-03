from __future__ import annotations

import os
from fastapi import FastAPI
from dotenv import load_dotenv

from app.utils.logging import setup_logging
from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router

load_dotenv()
setup_logging()

def create_app() -> FastAPI:
    app = FastAPI(title=os.getenv("APP_NAME", "estatein-ai-service"))

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")

    return app

app = create_app()
