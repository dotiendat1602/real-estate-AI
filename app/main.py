from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from .utils.logging import setup_logging
from .api.health import router as health_router
from .api.chat import router as chat_router
from .api.ingest import router as ingest_router
from .api.planning import router as planning_router

setup_logging()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("=" * 80)
    print(f"{os.getenv('APP_NAME', 'estatein-ai-service')} is starting...")
    print(f"Server: http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8001')}")
    
    # Khởi tạo vector store
    if _env_bool("AI_PRELOAD_VECTOR_STORES", True):
        print("Preloading vector stores...")
        from app.api import chat
        await chat.initialize_vector_store()
    else:
        print("Skipping vector store preload (AI_PRELOAD_VECTOR_STORES=false).")
    
    print("=" * 80)
    print("Application startup complete!")

    yield  # app chạy ở đây

    # shutdown
    print("\nApplication shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=os.getenv("APP_NAME", "estatein-ai-service"),
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")
    app.include_router(planning_router, prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
        reload=_env_bool("UVICORN_RELOAD", True),
    )
