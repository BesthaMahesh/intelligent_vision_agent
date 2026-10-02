from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Enforces unified interface for text generation and optional vision multimodal inference.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        """
        Generates text response from prompt and optional image payload.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks if provider is properly configured with valid credentials/endpoints.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
