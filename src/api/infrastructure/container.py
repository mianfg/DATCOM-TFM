"""Dependency injection container for API layer."""

from dependency_injector import containers, providers
from src.api.infrastructure.database import Database
from src.api.infrastructure.storage import FileStorage


class APIContainer(containers.DeclarativeContainer):
    """Dependency injection container for API layer."""

    config = providers.Configuration()

    # Database
    database = providers.Singleton(
        Database,
        database_url=config.database.url,
        async_database_url=config.database.async_url,
    )

    # File Storage
    file_storage = providers.Singleton(
        FileStorage,
        storage_path=config.storage.path,
    )


# Global container instance
_container: APIContainer | None = None


def init_container(config) -> APIContainer:
    """Initialize the global container."""
    global _container
    _container = APIContainer()
    _container.config.from_dict(config)
    return _container


def get_container() -> APIContainer:
    """Get the global container instance."""
    if _container is None:
        raise RuntimeError("Container not initialized. Call init_container() first.")
    return _container

