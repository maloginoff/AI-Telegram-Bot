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
            msg += f" Ориентировочное восстановление: {recovery_time}"
        super().__init__(msg, recoverable=False)


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

                async for chunk in gen:
                    yield chunk

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
                raise AiError("Превышено время ожидания ответа от AI. Попробуйте ещё раз.")

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
                        # Проверяем на quota exceeded в теле ответа
                        if "quota" in body.lower() or "rate" in body.lower():
                            raise KeyExhaustedException()
                        raise AiError(f"API error {resp.status}: {body[:200]}")

                    async for line in resp.content:
                        decoded = line.decode("utf-8", errors="ignore").strip()
                        if not decoded or not decoded.startswith("data: "):
                            continue
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    return
            except (KeyExhaustedException, KeyAuthError):
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
                        if "quota" in body.lower() or "rate" in body.lower():
                            raise KeyExhaustedException()
                        raise AiError(f"Gemini API error {resp.status}: {body[:200]}")

                    async for line in resp.content:
                        decoded = line.decode("utf-8", errors="ignore").strip()
                        if not decoded:
                            continue
                        if decoded.startswith("data: "):
                            decoded = decoded[6:]
                        try:
                            data = json.loads(decoded)
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
            except (KeyExhaustedException, KeyAuthError):
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
                # Gemini не поддерживает system role напрямую, преобразуем в user
                role = "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        # Gemini требует чередование ролей; убираем дублирующие подряд
        if not contents:
            return contents

        merged: list[dict] = [contents[0]]
        for c in contents[1:]:
            if c["role"] == merged[-1]["role"]:
                merged[-1]["parts"][0]["text"] += "\n" + c["parts"][0]["text"]
            else:
                merged.append(c)

        # Первое сообщение должно быть от user
        if merged and merged[0]["role"] == "model":
            merged.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})

        return merged


class KeyExhaustedException(Exception):
    pass


class KeyAuthError(Exception):
    pass


class ServerError(Exception):
    pass