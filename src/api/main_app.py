"""Main FastAPI application for the API layer."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import AppConfig
from src.api.infrastructure.container import init_container, get_container
from src.api.routers import collections, consolidation, documents, processing, rules, sensors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting DATCOM TFM API...")
    
    # Initialize configuration
    config = AppConfig()
    
    # Initialize API container
    api_config = {
        "database": {
            "url": config.database.url,
            "async_url": config.database.async_url,
        },
        "storage": {
            "path": config.storage.path,
        },
    }
    init_container(api_config)
    container = get_container()
    
    # Create database tables (for development)
    db = container.database()
    db.create_all()
    
    logger.info("✓ Database initialized")
    logger.info("✓ API ready")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")
    await db.close()
    logger.info("✓ Database connections closed")


# Create FastAPI app
app = FastAPI(
    title="DATCOM TFM API",
    description="API for document collection management and rule extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(processing.router)
app.include_router(rules.router)
app.include_router(sensors.router)
app.include_router(consolidation.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DATCOM TFM API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

