"""Configuration for the application."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    provider: str = Field(default="ollama", description="Embedding provider (openai or ollama)")
    model: str = Field(default="mxbai-embed-large", description="Embedding model name")
    base_url: str = Field(default="http://localhost:11434", description="Embedding API base URL")


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    collection_name: str = Field(default="plant_docs", description="Qdrant collection name")
    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: str | None = Field(default=None, description="Qdrant API key (for cloud)")


class DocumentLoaderConfig(BaseSettings):
    """Configuration for document loader."""

    model_config = SettingsConfigDict(env_prefix="DOCUMENT_", env_file=".env", extra="ignore")

    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")


class LLMConfig(BaseSettings):
    """Configuration for LLM."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: str = Field(default="deepseek", description="LLM provider (deepseek or ollama)")
    model: str = Field(default="deepseek-chat", description="LLM model name")
    base_url: str = Field(default="https://api.deepseek.com", description="LLM API base URL")
    api_key: str | None = Field(default=None, description="LLM API key")
    temperature: float = Field(default=0.2, description="LLM temperature")


class DatabaseConfig(BaseSettings):
    """Configuration for database."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5433, description="Database port")
    name: str = Field(default="datcom_db", description="Database name")
    user: str = Field(default="datcom_user", description="Database user")
    password: str = Field(default="datcom_password", description="Database password")

    @property
    def url(self) -> str:
        """Get synchronous database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """Get asynchronous database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class StorageConfig(BaseSettings):
    """Configuration for file storage."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_", env_file=".env", extra="ignore")

    path: str = Field(default="./storage", description="Base path for file storage")


class LangfuseConfig(BaseSettings):
    """Configuration for Langfuse tracing and monitoring."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    host: str = Field(default="http://localhost:3000", description="Langfuse host URL")
    public_key: str = Field(default="", description="Langfuse public key")
    secret_key: str = Field(default="", description="Langfuse secret key")


class GroundingConfig(BaseSettings):
    """Configuration for web grounding."""

    model_config = SettingsConfigDict(env_prefix="GROUNDING_", env_file=".env", extra="ignore")

    min_suggested_searches: int = Field(default=2, description="Minimum searches suggested to LLM")
    max_suggested_searches: int = Field(default=4, description="Maximum searches suggested to LLM")
    hard_limit: int = Field(default=4, description="Hard limit on searches performed (overrides LLM)")


class ProcessingConfig(BaseSettings):
    """Configuration for job processing."""

    model_config = SettingsConfigDict(env_prefix="PROCESSING_", env_file=".env", extra="ignore")

    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks in parallel")


class ConsolidationConfig(BaseSettings):
    """Configuration for rule consolidation."""

    model_config = SettingsConfigDict(env_prefix="CONSOLIDATION_", env_file=".env", extra="ignore")

    max_rules_per_batch: int = Field(default=50, description="Maximum rules to process in a single LLM call")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum valid confidence score")
    default_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Default confidence threshold")
    max_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Maximum confidence score")


class QueryConfig(BaseSettings):
    """Configuration for API query limits."""

    model_config = SettingsConfigDict(env_prefix="QUERY_", env_file=".env", extra="ignore")

    default_limit: int = Field(default=100, ge=1, description="Default number of items to return")
    max_limit: int = Field(default=1000, ge=1, description="Maximum number of items that can be requested")
    default_offset: int = Field(default=0, ge=0, description="Default pagination offset")


class PreviewConfig(BaseSettings):
    """Configuration for content preview lengths."""

    model_config = SettingsConfigDict(env_prefix="PREVIEW_", env_file=".env", extra="ignore")

    chunk_preview_length: int = Field(default=500, ge=100, description="Length of chunk content preview")
    source_preview_length: int = Field(default=500, ge=100, description="Length of source chunk preview in rules")


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document_loader: DocumentLoaderConfig = Field(default_factory=DocumentLoaderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
