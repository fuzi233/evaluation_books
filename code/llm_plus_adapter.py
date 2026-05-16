from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message


@dataclass
class ChatResponse:
    choices: list[Choice]


@dataclass
class EmbeddingItem:
    embedding: list[float]


@dataclass
class EmbeddingResponse:
    data: list[EmbeddingItem]


class ChatCompletions:
    def __init__(self, client: "LLMPlusCompatClient") -> None:
        self.client = client

    def create(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 4000, **_: Any) -> ChatResponse:
        text = self.client.generate_text(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return ChatResponse(choices=[Choice(message=Message(content=text))])


class Chat:
    def __init__(self, client: "LLMPlusCompatClient") -> None:
        self.completions = ChatCompletions(client)


class Embeddings:
    def create(self, *, input: str, model: str, **_: Any) -> EmbeddingResponse:
        return EmbeddingResponse(data=[EmbeddingItem(embedding=hash_embedding(input))])


class LLMPlusCompatClient:
    def __init__(self, *, api_key: str, api_url: str, model: str) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.chat = Chat(self)
        self.embeddings = Embeddings()

    def generate_text(self, *, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        system, user = split_messages(messages)
        body = {
            "model": model or self.model,
            "contents": [user],
            "system_prompt": system,
            "extra": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": float(os.environ.get("LLM_PLUS_TOP_P", "0.95")),
                "top_k": int(os.environ.get("LLM_PLUS_TOP_K", "40")),
                "include_thoughts": os.environ.get("LLM_PLUS_INCLUDE_THOUGHTS", "0").lower() in {"1", "true", "yes", "y"},
                "thinking_budget": int(os.environ.get("LLM_PLUS_THINKING_BUDGET", "1024")),
                "thinking_level": os.environ.get("LLM_PLUS_THINKING_LEVEL", "high"),
                "response_format": {"result": "str"},
            },
        }
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "4"))
        base_sleep = float(os.environ.get("LLM_RETRY_BASE_SLEEP", "1.0"))
        last_response_text = ""
        for attempt in range(max_retries + 1):
            with httpx.Client(timeout=180) as client:
                response = client.post(self.api_url, headers=headers, json=body)
            if response.status_code in {429, 502, 503, 504} and attempt < max_retries:
                last_response_text = response.text[:300]
                time.sleep(base_sleep * (2 ** attempt))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                preview = response.text[:300] or last_response_text
                raise RuntimeError(f"LLM-plus HTTP {response.status_code}: {preview!r}") from exc
            return extract_text(response.json())
        raise RuntimeError(f"LLM-plus request failed after retries: {last_response_text!r}")


def build_llm_client(args: Any, api_key: str) -> LLMPlusCompatClient:
    api_url = getattr(args, "llm_plus_api_url", "") or os.environ.get("LLM_PLUS_API_URL", "") or getattr(args, "base_url", "")
    model = getattr(args, "base_model", "") or os.environ.get("LLM_PLUS_MODEL", "gemini-3.1-pro")
    return LLMPlusCompatClient(api_key=api_key, api_url=api_url, model=model)


def split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_parts = []
    user_parts = []
    for message in messages:
        if message.get("role") == "system":
            system_parts.append(message.get("content", ""))
        else:
            user_parts.append(message.get("content", ""))
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def extract_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        raise RuntimeError(f"Unsupported llm-plus response type: {type(data).__name__}")
    if data.get("error"):
        raise RuntimeError(f"LLM-plus API error: {data}")
    try:
        value = data["choices"][0]["message"]["content"]
        if isinstance(value, str):
            return value
    except Exception:
        pass
    for key in ("text", "output_text", "content", "response", "result", "answer"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            try:
                return extract_text(value)
            except RuntimeError:
                pass
    for key in ("data", "results"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            try:
                return extract_text(value)
            except RuntimeError:
                pass
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    try:
                        return extract_text(item)
                    except RuntimeError:
                        pass
    candidates = data.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if text:
            return text
    raise RuntimeError(f"Could not extract text from llm-plus response: {data}")


def hash_embedding(text: str, dim: int = 1536) -> list[float]:
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dim
        vector[index] += 1.0 if digest[0] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0 else [value / norm for value in vector]
