"""LLM providers for Agent layer - moved from src/infrastructure/llm/"""

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek


class DeepSeekLLMProvider:
    """DeepSeek-based LLM provider (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.2,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self._llm = None

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = ChatDeepSeek(
                model=self.model,
                api_key="sk-a2ab49c78aeb4a4592ab3ab724f4ea84",  # self.api_key,
                temperature=self.temperature,
            )
        return self._llm
