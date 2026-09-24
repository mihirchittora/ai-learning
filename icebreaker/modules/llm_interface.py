"""Hugging Face embeddings and LLM setup for the Icebreaker Bot."""

import os
from typing import Any, Generator, Optional

from huggingface_hub import InferenceClient, get_token
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import PrivateAttr

import config


def _get_huggingface_token() -> str:
    """Return the configured token without exposing it in logs or responses."""

    token = os.getenv(config.HF_TOKEN_ENV_VAR) or get_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export it before starting the application."
        )
    return token


class HuggingFaceLLM(CustomLLM):
    """Small LlamaIndex adapter around Hugging Face's current chat API."""

    model_name: str = config.LLM_MODEL_ID
    temperature: float = config.TEMPERATURE
    max_new_tokens: int = config.MAX_NEW_TOKENS
    _client: InferenceClient = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        resolved_model = model_name or config.LLM_MODEL_ID
        super().__init__(
            model_name=resolved_model,
            temperature=config.TEMPERATURE if temperature is None else temperature,
            max_new_tokens=(
                config.MAX_NEW_TOKENS
                if max_new_tokens is None
                else max_new_tokens
            ),
            callback_manager=callback_manager or CallbackManager([]),
            **kwargs,
        )
        self._client = InferenceClient(
            model=resolved_model,
            provider="auto",
            token=_get_huggingface_token(),
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            model_name=self.model_name,
        )

    def _generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face inference failed. Check HF_TOKEN, the model, and network access."
            ) from exc

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Hugging Face returned an unexpected response.") from exc

        if not answer:
            raise RuntimeError("Hugging Face returned an empty response.")
        return str(answer).strip()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=self._generate(prompt))

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        answer = self._generate(prompt)

        def generate() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=answer, delta=answer)

        return generate()


def create_huggingface_embedding() -> HuggingFaceEmbedding:
    """Create the local Hugging Face sentence-transformer embedding model."""

    return HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_ID)


def create_huggingface_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> HuggingFaceLLM:
    """Create a LlamaIndex-compatible Hugging Face LLM."""

    return HuggingFaceLLM(
        model_name=config.LLM_MODEL_ID,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def change_llm_model(new_model_id: str) -> None:
    """Change the model used by subsequently created LLM instances."""

    if not new_model_id or not new_model_id.strip():
        raise ValueError("The Hugging Face model ID cannot be empty.")
    config.LLM_MODEL_ID = new_model_id.strip()
