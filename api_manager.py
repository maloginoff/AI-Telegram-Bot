import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

from config import Config
from database import Database

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    raw_key: str
    key_hash: str
    provider: str
    status: str = "active"
    last_exhausted: float = 0.0


class ApiKeyManager:
    def __init__(self, config: Config, database: Database) -> None:
        self._config = config
        self._db = database
        self._keys: dict[str, list[KeyState]] = {"openrouter": [], "gemini": []}
        self._current_index: dict[str, int] = {"openrouter": 0, "gemini": 0}
        self._locks: dict[str, asyncio.Lock] = {
            "openrouter": asyncio.Lock(),
            "gemini": asyncio.Lock(),
        }
        self._recovery_task: asyncio.Task | None = None

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def initialize(self) -> None:
        for raw_key in self._config.api.openrouter_keys:
            ks = KeyState(
                raw_key=raw_key,
                key_hash=self._hash_key(raw_key),
                provider="openrouter",
            )
            self._keys["openrouter"].append(ks)
            await self._db.upsert_api_key("openrouter", ks.key_hash)

        for raw_key in self._config.api.gemini_keys:
            ks = KeyState(
                raw_key=raw_key,
                key_hash=self._hash_key(raw_key),
                provider="gemini",
            )
            self._keys["gemini"].append(ks)
            await self._db.upsert_api_key("gemini", ks.key_hash)

        # Синхронизируем статусы из БД
        for provider in ("openrouter", "gemini"):
            db_keys = await self._db.get_api_keys(provider)
            db_map = {k["key_hash"]: k["status"] for k in db_keys}
            for ks in self._keys[provider]:
                if ks.key_hash in db_map:
                    ks.status = db_map[ks.key_hash]

        total = sum(len(v) for v in self._keys.values())
        active = sum(1 for keys in self._keys.values() for k in keys if k.status == "active")
        logger.info("API keys loaded: %d total, %d active", total, active)

    def start_recovery_loop(self) -> None:
        self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop_recovery_loop(self) -> None:
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

    async def _recovery_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)  # проверяем каждые 5 минут
                cooldown = self._config.api.key_cooldown_minutes
                for provider in ("openrouter", "gemini"):
                    recovered = await self._db.reset_exhausted_keys(provider, cooldown)
                    if recovered > 0:
                        # Обновляем in-memory состояние
                        async with self._locks[provider]:
                            now = time.time()
                            cooldown_sec = cooldown * 60
                            for ks in self._keys[provider]:
                                if ks.status == "exhausted" and (now - ks.last_exhausted) >= cooldown_sec:
                                    ks.status = "active"
                                    logger.info(
                                        "Key %s (%s) recovered to active",
                                        ks.key_hash, provider,
                                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Recovery loop error: %s", e)
                await asyncio.sleep(60)

    async def get_key(self, provider: str) -> KeyState | None:
        async with self._locks[provider]:
            keys = self._keys.get(provider, [])
            if not keys:
                return None

            active_keys = [k for k in keys if k.status == "active"]
            if not active_keys:
                return None

            idx = self._current_index[provider] % len(active_keys)
            self._current_index[provider] = idx + 1
            return active_keys[idx]

    async def mark_exhausted(self, key_hash: str, provider: str) -> None:
        async with self._locks[provider]:
            for ks in self._keys[provider]:
                if ks.key_hash == key_hash:
                    ks.status = "exhausted"
                    ks.last_exhausted = time.time()
                    await self._db.update_key_status(key_hash, "exhausted")
                    logger.warning("Key %s (%s) marked exhausted", key_hash, provider)
                    break

    async def mark_error(self, key_hash: str, provider: str) -> None:
        async with self._locks[provider]:
            for ks in self._keys[provider]:
                if ks.key_hash == key_hash:
                    ks.status = "error"
                    await self._db.update_key_status(key_hash, "error")
                    logger.error("Key %s (%s) marked error (auth failure)", key_hash, provider)
                    break

    async def mark_active(self, key_hash: str, provider: str) -> None:
        async with self._locks[provider]:
            for ks in self._keys[provider]:
                if ks.key_hash == key_hash:
                    ks.status = "active"
                    await self._db.update_key_status(key_hash, "active")
                    logger.info("Key %s (%s) manually set to active", key_hash, provider)
                    break

    async def record_usage(self, key_hash: str) -> None:
        await self._db.increment_key_requests(key_hash)

    async def get_all_keys_status(self, provider: str | None = None) -> list[dict]:
        result = []
        providers = [provider] if provider else ["openrouter", "gemini"]
        for p in providers:
            db_keys = await self._db.get_api_keys(p)
            db_map = {k["key_hash"]: k for k in db_keys}
            for ks in self._keys.get(p, []):
                db_info = db_map.get(ks.key_hash, {})
                result.append({
                    "key_hash": ks.key_hash,
                    "provider": ks.provider,
                    "status": ks.status,
                    "total_requests": db_info.get("total_requests", 0),
                    "exhausted_count": db_info.get("exhausted_count", 0),
                    "last_used": db_info.get("last_used"),
                    "last_exhausted": db_info.get("last_exhausted"),
                })
        return result

    async def has_active_keys(self, provider: str) -> bool:
        async with self._locks[provider]:
            return any(k.status == "active" for k in self._keys.get(provider, []))

    async def get_recovery_time(self, provider: str) -> str | None:
        return await self._db.get_earliest_exhausted_recovery(
            provider, self._config.api.key_cooldown_minutes
        )

    def get_providers_with_keys(self) -> list[str]:
        return [p for p, keys in self._keys.items() if keys]