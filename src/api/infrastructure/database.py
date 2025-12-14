"""Database infrastructure for API layer."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from loguru import logger

from src.api.domain.models import Base


class Database:
    """Database connection manager."""

    def __init__(self, database_url: str, async_database_url: str):
        """
        Initialize database connection.

        Args:
            database_url: Synchronous database URL (for migrations)
            async_database_url: Asynchronous database URL (for FastAPI)
        """
        self.database_url = database_url
        self.async_database_url = async_database_url

        # Sync engine for migrations
        self.sync_engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=20,  # Base connection pool size (increased for parallel tasks)
            max_overflow=10,  # Additional connections under load
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_timeout=30,  # Timeout waiting for connection (seconds)
        )
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False,
        )

        # Async engine for FastAPI (optimized for concurrent operations)
        self.async_engine = create_async_engine(
            async_database_url,
            echo=False,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=20,  # Base connection pool size (increased for parallel tasks)
            max_overflow=10,  # Additional connections under load
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_timeout=30,  # Timeout waiting for connection (seconds)
        )
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        logger.info(
            f"Database connection pool configured: "
            f"pool_size=20, max_overflow=10, pool_recycle=3600s"
        )

    def create_all(self):
        """Create all tables (for development only)."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.sync_engine)
        logger.info("✓ Database tables created")

    def drop_all(self):
        """Drop all tables (for development only)."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.sync_engine)
        logger.info("✓ Database tables dropped")

    def get_sync_session(self) -> Session:
        """Get a synchronous database session."""
        return self.sync_session_factory()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self):
        """Close database connections."""
        await self.async_engine.dispose()
        self.sync_engine.dispose()
        logger.info("✓ Database connections closed")


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session."""
    from src.api.infrastructure.container import get_container

    container = get_container()
    db = container.database()

    async with db.get_async_session() as session:
        yield session

