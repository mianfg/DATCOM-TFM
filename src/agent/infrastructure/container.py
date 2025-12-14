"""Dependency injection container for Agent layer."""

from dependency_injector import containers, providers


class AgentContainer(containers.DeclarativeContainer):
    """Dependency injection container for Agent layer."""

    config = providers.Configuration()

    # Infrastructure - Embedding Provider
    from src.agent.infrastructure.embeddings import OllamaEmbeddingProvider

    embedding_provider = providers.Singleton(
        OllamaEmbeddingProvider,
        model=config.embedding.model,
        base_url=config.embedding.base_url,
    )

    # Infrastructure - Vector Store
    from src.agent.infrastructure.vector_stores import QdrantVectorStore

    vector_store = providers.Factory(
        QdrantVectorStore,
        embedding_provider=embedding_provider,
        collection_name=config.vector_store.collection_name,
        url=config.vector_store.url,
        api_key=config.vector_store.api_key,
    )

    # Infrastructure - Document Loader
    from src.agent.infrastructure.document_loaders import DoclingDocumentLoader

    document_loader = providers.Factory(
        DoclingDocumentLoader,
        chunk_size=config.document_loader.chunk_size,
        chunk_overlap=config.document_loader.chunk_overlap,
    )

    # Infrastructure - LLM Provider
    from src.agent.infrastructure.llm import DeepSeekLLMProvider

    llm_provider = providers.Singleton(
        DeepSeekLLMProvider,
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
    )

    # Application - Use Cases
    from src.agent.application.consolidation_use_case import RuleConsolidationUseCase
    from src.agent.application.extraction_use_case import RuleExtractionUseCase

    rule_extraction_use_case = providers.Factory(
        RuleExtractionUseCase,
        document_loader=document_loader,
        vector_store=vector_store,
        llm_provider=llm_provider,
        min_suggested_searches=config.grounding.min_suggested_searches,
        max_suggested_searches=config.grounding.max_suggested_searches,
        hard_limit_searches=config.grounding.hard_limit,
    )

    rule_consolidation_use_case = providers.Factory(
        RuleConsolidationUseCase,
        llm_provider=llm_provider,
    )


# Global container instance
_agent_container: AgentContainer | None = None


def get_agent_container() -> AgentContainer:
    """Get the global agent container instance."""
    global _agent_container
    if _agent_container is None:
        from src.config import AppConfig

        config = AppConfig()
        _agent_container = AgentContainer()
        _agent_container.config.from_pydantic(config)
    return _agent_container
