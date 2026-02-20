import logging
import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, CallbackQuery, Update

from database import Database

logger = logging.getLogger(__name__)


class UserRegistrationMiddleware(BaseMiddleware):
    def __init__(self, database: Database) -> None:
        self._db = database

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        user = event.from_user
        if user and not user.is_bot:
            await self._db.upsert_user(
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
            )
            if await self._db.is_banned(user.id):
                await event.answer("⛔ Вы заблокированы.")
                return None
        return await handler(event, data)


class CallbackRegistrationMiddleware(BaseMiddleware):
    def __init__(self, database: Database) -> None:
        self._db = database

    async def __call__(
        self,
        handler: Callable[[CallbackQuery, dict[str, Any]], Awaitable[Any]],
        event: CallbackQuery,
        data: dict[str, Any],
    ) -> Any:
        user = event.from_user
        if user and not user.is_bot:
            await self._db.upsert_user(
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
            )
            if await self._db.is_banned(user.id):
                await event.answer("⛔ Вы заблокированы.", show_alert=True)
                return None
        return await handler(event, data)


class ThrottleMiddleware(BaseMiddleware):
    def __init__(self, rate_limit: float = 1.0) -> None:
        self._rate_limit = rate_limit
        self._user_last: dict[int, float] = {}

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        user = event.from_user
        if not user:
            return await handler(event, data)

        now = time.monotonic()
        last = self._user_last.get(user.id, 0.0)

        if now - last < self._rate_limit:
            return None

        self._user_last[user.id] = now
        return await handler(event, data)


class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        start = time.monotonic()

        if isinstance(event, Message) and event.from_user:
            logger.debug(
                "Message from %s (%d): %s",
                event.from_user.username or event.from_user.first_name,
                event.from_user.id,
                (event.text or "")[:100],
            )
        elif isinstance(event, CallbackQuery) and event.from_user:
            logger.debug(
                "Callback from %s (%d): %s",
                event.from_user.username or event.from_user.first_name,
                event.from_user.id,
                event.data,
            )

        try:
            result = await handler(event, data)
            elapsed = (time.monotonic() - start) * 1000
            logger.debug("Handler completed in %.1f ms", elapsed)
            return result
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "Handler error after %.1f ms: %s: %s",
                elapsed, type(e).__name__, e,
            )
            raise