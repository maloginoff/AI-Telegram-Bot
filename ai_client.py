import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import aiohttp

from api_manager import ApiKeyManager, KeyState
from config import Config

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"


class AiError(Exception):
    def __init__(self, message: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class AllKeysExhaustedError(AiError):
    def __init__(self, recovery_time: str | None = None) -> None:
        self.recovery_time = recovery_time
        msg = "Все API-ключи временно исчерпаны."
        if recovery_time:
            msg += f"\n⏱ Ориентировочное восстановление: {recovery_time}"
        super().__init__(msg, recoverable=False)


class KeyExhaustedException(Exception):
    pass


class KeyAuthError(Exception):
    pass


class ServerError(Exception):
    pass


class AiClient:
    def __init__(self, config: Config, key_manager: ApiKeyManager) -> None:
        self._config = config
        self._key_manager = key_manager
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._config.api.request_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("AiClient session not started")
        return self._session

    async def stream_response(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
    ) -> AsyncGenerator[str, None]:
        max_key_attempts = 5
        for attempt in range(max_key_attempts):
            key = await self._key_manager.get_key(provider)
            if key is None:
                recovery = await self._key_manager.get_recovery_time(provider)
                raise AllKeysExhaustedError(recovery)

            try:
                if provider == "openrouter":
                    gen = self._stream_openrouter(messages, model, key)
                else:
                    gen = self._stream_gemini(messages, model, key)

                collected = False
                async for chunk in gen:
                    collected = True
                    yield chunk

                if collected:
                    await self._key_manager.record_usage(key.key_hash)
                return

            except KeyExhaustedException:
                await self._key_manager.mark_exhausted(key.key_hash, provider)
                logger.warning(
                    "Key %s exhausted (attempt %d/%d), rotating...",
                    key.key_hash, attempt + 1, max_key_attempts,
                )
                continue

            except KeyAuthError:
                await self._key_manager.mark_error(key.key_hash, provider)
                logger.error("Key %s auth error, disabling", key.key_hash)
                continue

            except ServerError as e:
                logger.warning("Server error: %s, retrying...", e)
                await asyncio.sleep(5)
                continue

            except asyncio.TimeoutError:
                logger.warning("Request timeout for key %s", key.key_hash)
                raise AiError("⏱ Превышено время ожидания ответа от AI. Попробуйте ещё раз.")

        recovery = await self._key_manager.get_recovery_time(provider)
        raise AllKeysExhaustedError(recovery)

    def _check_error_in_body(self, text: str) -> None:
        """Проверяет тело ответа на ошибки, которые приходят со статусом 200."""
        lower = text.lower()
        rate_keywords = [
            "rate_limit", "rate limit", "quota exceeded", "resource_exhausted",
            "too many requests", "limit reached", "credits", "insufficient",
        ]
        auth_keywords = ["invalid api key", "invalid_api_key", "unauthorized", "forbidden"]

        for kw in rate_keywords:
            if kw in lower:
                raise KeyExhaustedException()
        for kw in auth_keywords:
            if kw in lower:
                raise KeyAuthError()

    async def _stream_openrouter(
        self,
        messages: list[dict[str, str]],
        model: str,
        key: KeyState,
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {key.raw_key}",
            "HTTP-Referer": "https://github.com/ai-telegram-bot",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": 4096,
        }

        max_retries = 3
        for retry in range(max_retries):
            try:
                async with self.session.post(
                    OPENROUTER_URL, json=payload, headers=headers
                ) as resp:
                    if resp.status == 429:
                        raise KeyExhaustedException()
                    if resp.status in (401, 403):
                        raise KeyAuthError()
                    if resp.status >= 500:
                        if retry < max_retries - 1:
                            await asyncio.sleep(5)
                            continue
                        raise ServerError(f"Server returned {resp.status}")

                    if resp.status != 200:
                        body = await resp.text()
                        self._check_error_in_body(body)
                        raise AiError(f"API error {resp.status}: {body[:200]}")

                    # Проверяем content-type — если не stream, читаем целиком
                    content_type = resp.headers.get("content-type", "")
                    if "text/event-stream" not in content_type and "stream" not in content_type:
                        body = await resp.text()
                        self._check_error_in_body(body)
                        # Может быть обычный JSON-ответ
                        try:
                            data = json.loads(body)
                            error = data.get("error", {})
                            if error:
                                error_msg = error.get("message", str(error))
                                self._check_error_in_body(error_msg)
                                raise AiError(f"API error: {error_msg[:200]}")
                            # Извлекаем контент из не-stream ответа
                            content = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            if content:
                                yield content
                                return
                        except json.JSONDecodeError:
                            pass
                        return

                    # Stream-чтение
                    buffer = ""
                    async for raw_chunk in resp.content.iter_any():
                        buffer += raw_chunk.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith("data: "):
                                data_str = line[6:]
                            elif line.startswith("data:"):
                                data_str = line[5:]
                            else:
                                # Может быть JSON-ошибка без SSE-обёртки
                                self._check_error_in_body(line)
                                continue

                            if data_str.strip() == "[DONE]":
                                return

                            try:
                                data = json.loads(data_str)
                                # Проверяем ошибки внутри SSE
                                if "error" in data:
                                    error_msg = data["error"]
                                    if isinstance(error_msg, dict):
                                        error_msg = error_msg.get("message", str(error_msg))
                                    self._check_error_in_body(str(error_msg))
                                    raise AiError(f"Stream error: {str(error_msg)[:200]}")

                                delta = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    yield delta
                            except json.JSONDecodeError:
                                continue
                    return

            except (KeyExhaustedException, KeyAuthError, AiError):
                raise
            except aiohttp.ClientError as e:
                if retry < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                raise ServerError(str(e))

    async def _stream_gemini(
        self,
        messages: list[dict[str, str]],
        model: str,
        key: KeyState,
    ) -> AsyncGenerator[str, None]:
        url = GEMINI_URL.format(model=model)
        params = {"key": key.raw_key, "alt": "sse"}

        contents = self._convert_messages_to_gemini(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 4096,
                "temperature": 0.7,
            },
        }

        max_retries = 3
        for retry in range(max_retries):
            try:
                async with self.session.post(
                    url, json=payload, params=params
                ) as resp:
                    if resp.status == 429:
                        raise KeyExhaustedException()
                    if resp.status in (401, 403):
                        raise KeyAuthError()
                    if resp.status >= 500:
                        if retry < max_retries - 1:
                            await asyncio.sleep(5)
                            continue
                        raise ServerError(f"Gemini server returned {resp.status}")

                    if resp.status != 200:
                        body = await resp.text()
                        self._check_error_in_body(body)
                        raise AiError(f"Gemini API error {resp.status}: {body[:200]}")

                    buffer = ""
                    async for raw_chunk in resp.content.iter_any():
                        buffer += raw_chunk.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith("data: "):
                                line = line[6:]
                            elif line.startswith("data:"):
                                line = line[5:]

                            self._check_error_in_body(line)

                            try:
                                data = json.loads(line)
                                if "error" in data:
                                    error_msg = data["error"]
                                    if isinstance(error_msg, dict):
                                        error_msg = error_msg.get("message", str(error_msg))
                                    self._check_error_in_body(str(error_msg))
                                    raise AiError(f"Gemini error: {str(error_msg)[:200]}")

                                candidates = data.get("candidates", [])
                                if not candidates:
                                    continue
                                parts = (
                                    candidates[0]
                                    .get("content", {})
                                    .get("parts", [])
                                )
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                continue
                    return

            except (KeyExhaustedException, KeyAuthError, AiError):
                raise
            except aiohttp.ClientError as e:
                if retry < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                raise ServerError(str(e))

    @staticmethod
    def _convert_messages_to_gemini(
        messages: list[dict[str, str]],
    ) -> list[dict]:
        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        if not contents:
            return contents

        merged: list[dict] = [contents[0]]
        for c in contents[1:]:
            if c["role"] == merged[-1]["role"]:
                merged[-1]["parts"][0]["text"] += "\n" + c["parts"][0]["text"]
            else:
                merged.append(c)

        if merged and merged[0]["role"] == "model":
            merged.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})

        return merged