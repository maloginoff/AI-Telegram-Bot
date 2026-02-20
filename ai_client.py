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


def _is_rate_limit_error(error_obj: dict | str) -> bool:
    """Проверяет является ли ошибка rate limit / quota exceeded."""
    if isinstance(error_obj, dict):
        text = json.dumps(error_obj).lower()
    else:
        text = str(error_obj).lower()

    rate_phrases = [
        "rate_limit_exceeded",
        "rate limit",
        "quota exceeded",
        "resource_exhausted",
        "resource has been exhausted",
        "too many requests",
        "limit reached",
        "insufficient_quota",
        "exceeded your current quota",
        "requests per minute",
        "tokens per minute",
    ]
    return any(phrase in text for phrase in rate_phrases)


def _is_auth_error(error_obj: dict | str) -> bool:
    """Проверяет является ли ошибка авторизационной."""
    if isinstance(error_obj, dict):
        text = json.dumps(error_obj).lower()
    else:
        text = str(error_obj).lower()

    auth_phrases = [
        "invalid api key",
        "invalid_api_key",
        "api key not valid",
        "api_key_invalid",
        "permission denied",
        "authentication failed",
    ]
    return any(phrase in text for phrase in auth_phrases)


def _classify_error(error_obj: dict | str) -> None:
    """Бросает нужное исключение если ошибка распознана."""
    if _is_rate_limit_error(error_obj):
        raise KeyExhaustedException()
    if _is_auth_error(error_obj):
        raise KeyAuthError()


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
                    logger.debug(
                        "OpenRouter response: status=%d, content-type=%s, key=%s",
                        resp.status, resp.headers.get("content-type", ""), key.key_hash,
                    )

                    if resp.status == 429:
                        body = await resp.text()
                        logger.warning("OpenRouter 429: %s", body[:500])
                        raise KeyExhaustedException()

                    if resp.status in (401, 403):
                        body = await resp.text()
                        logger.warning("OpenRouter %d: %s", resp.status, body[:500])
                        raise KeyAuthError()

                    if resp.status >= 500:
                        body = await resp.text()
                        logger.warning("OpenRouter %d: %s", resp.status, body[:500])
                        if retry < max_retries - 1:
                            await asyncio.sleep(5)
                            continue
                        raise ServerError(f"Server returned {resp.status}")

                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("OpenRouter %d: %s", resp.status, body[:500])
                        _classify_error(body)
                        raise AiError(f"API error {resp.status}: {body[:200]}")

                    # Проверяем content-type
                    content_type = resp.headers.get("content-type", "")

                    if "text/event-stream" not in content_type and "stream" not in content_type:
                        # Не-stream ответ — читаем целиком
                        body = await resp.text()
                        logger.debug("OpenRouter non-stream body: %s", body[:1000])

                        try:
                            data = json.loads(body)

                            # Проверяем ошибку в JSON
                            if "error" in data:
                                error_obj = data["error"]
                                logger.warning("OpenRouter error in body: %s", error_obj)
                                _classify_error(error_obj)
                                error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                                raise AiError(f"API error: {error_msg[:200]}")

                            # Извлекаем контент из обычного ответа
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                                if content:
                                    yield content
                                    return
                        except json.JSONDecodeError:
                            logger.warning("OpenRouter non-JSON body: %s", body[:500])
                        return

                    # SSE stream
                    buffer = ""
                    async for raw_chunk in resp.content.iter_any():
                        buffer += raw_chunk.decode("utf-8", errors="ignore")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            # Извлекаем data из SSE
                            if line.startswith("data: "):
                                data_str = line[6:]
                            elif line.startswith("data:"):
                                data_str = line[5:]
                            elif line.startswith(":"):
                                continue
                            else:
                                continue

                            data_str = data_str.strip()
                            if data_str == "[DONE]":
                                return
                            if not data_str:
                                continue

                            try:
                                data = json.loads(data_str)

                                # Ошибка внутри SSE
                                if "error" in data:
                                    error_obj = data["error"]
                                    logger.warning("OpenRouter stream error: %s", error_obj)
                                    _classify_error(error_obj)
                                    error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                                    raise AiError(f"Stream error: {error_msg[:200]}")

                                choices = data.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content

                            except json.JSONDecodeError:
                                continue
                    return

            except (KeyExhaustedException, KeyAuthError, AiError):
                raise
            except aiohttp.ClientError as e:
                logger.warning("OpenRouter connection error: %s", e)
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
                    logger.debug(
                        "Gemini response: status=%d, content-type=%s, key=%s",
                        resp.status, resp.headers.get("content-type", ""), key.key_hash,
                    )

                    if resp.status == 429:
                        body = await resp.text()
                        logger.warning("Gemini 429: %s", body[:500])
                        raise KeyExhaustedException()

                    if resp.status in (401, 403):
                        body = await resp.text()
                        logger.warning("Gemini %d: %s", resp.status, body[:500])
                        raise KeyAuthError()

                    if resp.status >= 500:
                        body = await resp.text()
                        logger.warning("Gemini %d: %s", resp.status, body[:500])
                        if retry < max_retries - 1:
                            await asyncio.sleep(5)
                            continue
                        raise ServerError(f"Gemini server returned {resp.status}")

                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Gemini %d: %s", resp.status, body[:500])

                        # Парсим JSON-ошибку
                        try:
                            data = json.loads(body)
                            if "error" in data:
                                error_obj = data["error"]
                                logger.warning("Gemini error: %s", error_obj)
                                _classify_error(error_obj)
                                error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                                raise AiError(f"Gemini error: {error_msg[:200]}")
                        except json.JSONDecodeError:
                            pass

                        _classify_error(body)
                        raise AiError(f"Gemini API error {resp.status}: {body[:200]}")

                    # SSE stream
                    buffer = ""
                    async for raw_chunk in resp.content.iter_any():
                        buffer += raw_chunk.decode("utf-8", errors="ignore")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            # Убираем SSE-префикс
                            if line.startswith("data: "):
                                line = line[6:]
                            elif line.startswith("data:"):
                                line = line[5:]
                            elif line.startswith(":"):
                                continue

                            line = line.strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)

                                # Проверяем ошибку
                                if "error" in data:
                                    error_obj = data["error"]
                                    logger.warning("Gemini stream error: %s", error_obj)
                                    _classify_error(error_obj)
                                    error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                                    raise AiError(f"Gemini error: {error_msg[:200]}")

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
                logger.warning("Gemini connection error: %s", e)
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