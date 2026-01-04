from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from app.utils.logging import setup_logging
from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("=" * 80)
    print(f"{os.getenv('APP_NAME', 'estatein-ai-service')} is starting...")
    print(f"Server: http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8001')}")
    
    # Khởi tạo vector store
    print("Initializing vector store...")
    from app.api import chat, ingest
    await chat.initialize_vector_store()
    await ingest.initialize_vector_store()
    
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

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
        reload=True,
    )
