"""Langfuse client initialization and helper functions."""

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from loguru import logger

from src.config import AppConfig

# Load .env file into os.environ at module import
# This ensures Langfuse can read from environment variables
load_dotenv(override=False)  # Don't override existing env vars


class LangfuseClientManager:
    """Manages Langfuse client and callback handler creation."""

    def __init__(self):
        """Initialize with app config."""
        self.config = AppConfig()
        self._client = None
        self._verified = False

    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled in configuration."""
        return self.config.langfuse.enabled

    def get_client(self):
        """
        Get Langfuse client instance.

        Returns authenticated client or None if disabled/invalid.

        Note: Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
        to be set in environment (loaded from .env automatically at module import).
        """
        if not self.is_enabled():
            return None

        if self._client is None:
            try:
                # Initialize client using environment variables
                # Environment variables are loaded from .env at module import via load_dotenv()
                self._client = get_client()
                logger.debug("Langfuse client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {e}")
                return None

        return self._client

    def verify_auth(self) -> bool:
        """
        Verify Langfuse authentication.

        Returns True if authenticated successfully, False otherwise.
        """
        if not self.is_enabled():
            logger.info("Langfuse is disabled in configuration")
            return False

        if self._verified:
            return True

        client = self.get_client()
        if not client:
            return False

        try:
            if client.auth_check():
                logger.info("✅ Langfuse client is authenticated and ready!")
                self._verified = True
                return True
            else:
                logger.error("❌ Langfuse authentication failed. Please check your credentials and host.")
                return False
        except Exception as e:
            logger.error(f"❌ Langfuse authentication error: {e}")
            return False

    def get_callback_handler(self) -> CallbackHandler | None:
        """
        Create a Langfuse callback handler for LangChain/LangGraph.

        Note: Session ID, metadata, and tags should be set using propagate_attributes()
        when invoking the chain/workflow, not in the handler constructor.

        Returns:
            CallbackHandler instance or None if Langfuse is disabled/invalid
        """
        if not self.is_enabled():
            return None

        # Verify auth on first use
        if not self._verified and not self.verify_auth():
            return None

        try:
            # Create handler without parameters (credentials come from env vars)
            handler = CallbackHandler()
            logger.debug("Created Langfuse callback handler")
            return handler
        except Exception as e:
            logger.error(f"Failed to create Langfuse callback handler: {e}")
            return None


# Singleton instance
_langfuse_manager: LangfuseClientManager | None = None


def get_langfuse_manager() -> LangfuseClientManager:
    """Get singleton Langfuse manager instance."""
    global _langfuse_manager
    if _langfuse_manager is None:
        _langfuse_manager = LangfuseClientManager()
    return _langfuse_manager


def verify_langfuse_connection() -> bool:
    """
    Verify Langfuse connection and authentication.

    Call this at startup to ensure Langfuse is properly configured.

    Returns:
        True if Langfuse is enabled and authenticated, False otherwise.
    """
    manager = get_langfuse_manager()
    return manager.verify_auth()
