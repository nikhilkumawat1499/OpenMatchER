from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field


class AdjudicationRequest(BaseModel):
    record_a: dict[str, Any]
    record_b: dict[str, Any]
    features: dict[str, float]
    blocking_metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_similarity: float | None = None
    candidate_context: dict[str, Any] = Field(default_factory=dict)


class AdjudicationResult(BaseModel):
    match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_level: str = Field(default="medium", pattern="^(low|medium|high)$")


class LLMProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMProvider(ABC):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    base_url = "https://api.openai.com/v1"
    max_parse_attempts = 2

    async def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult:
        prompt = (
            "You are an entity-resolution adjudicator. Determine whether Entity A and Entity B "
            "represent the same real-world entity. Consider feature scores, blocking metadata, "
            "embedding similarity, and candidate context. Return only valid JSON matching this schema: "
            '{"match": true, "confidence": 0.95, "reasoning": "...", "risk_level": "low"}.\n'
            f"Entity A: {request.record_a}\n"
            f"Entity B: {request.record_b}\n"
            f"Feature Scores: {request.features}\n"
            f"Blocking Metadata: {request.blocking_metadata}\n"
            f"Embedding Similarity: {request.embedding_similarity}\n"
            f"Candidate Context: {request.candidate_context}"
        )
        try:
            last_error: Exception | None = None
            async with httpx.AsyncClient(timeout=30) as client:
                messages = [{"role": "user", "content": prompt}]
                for _ in range(self.max_parse_attempts):
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    try:
                        return AdjudicationResult.model_validate_json(content)
                    except ValueError as exc:
                        last_error = exc
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": "The previous response did not match the required schema. Return only valid JSON with match, confidence, reasoning, and risk_level.",
                            }
                        )
            raise LLMProviderError(self.__class__.__name__, "LLM provider returned an invalid response.") from last_error
        except httpx.HTTPStatusError as exc:
            provider_status = exc.response.status_code
            if provider_status == 401:
                message = "LLM provider rejected the API key."
            elif provider_status == 429:
                message = "LLM provider rate limit or quota was reached."
            else:
                message = f"LLM provider returned HTTP {provider_status}."
            raise LLMProviderError(self.__class__.__name__, message) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(self.__class__.__name__, "Could not reach the LLM provider.") from exc
        except KeyError as exc:
            raise LLMProviderError(self.__class__.__name__, "LLM provider returned an invalid response.") from exc


class OpenRouterProvider(OpenAICompatibleProvider):
    base_url = "https://openrouter.ai/api/v1"


class OllamaProvider(OpenAICompatibleProvider):
    base_url = "http://ollama:11434/v1"


class AnthropicProvider(OpenAICompatibleProvider):
    base_url = "https://api.anthropic.com/v1"


class GeminiProvider(OpenAICompatibleProvider):
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"


SUPPORTED_PROVIDERS = {
    "openai": OpenAICompatibleProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def provider_from_name(name: str, api_key: str, model: str) -> LLMProvider:
    try:
        return SUPPORTED_PROVIDERS[name.lower()](api_key, model)
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {name}") from exc
