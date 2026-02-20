import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from config import load_config, Config
from database import Database
from api_manager import ApiKeyManager
from ai_client import AiClient
from context_manager import ContextManager
from middlewares import (
    UserRegistrationMiddleware,
    CallbackRegistrationMiddleware,
    ThrottleMiddleware,
    LoggingMiddleware,
)
from handlers import user, admin, callbacks


def setup_logging(config: Config) -> None:
    log_dir = Path(config.log.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        config.log.file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.log.level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


async def on_startup(
    bot: Bot,
    config: Config,
    database: Database,
    key_manager: ApiKeyManager,
    ai_client: AiClient,
) -> None:
    await database.connect()
    await key_manager.initialize()
    key_manager.start_recovery_loop()
    await ai_client.start()

    bot_info = await bot.get_me()
    logger.info("Bot started: @%s (ID: %d)", bot_info.username, bot_info.id)

    for admin_id in config.admin_ids:
        try:
            await bot.send_message(admin_id, "✅ Бот запущен и готов к работе.")
        except Exception:
            pass


async def on_shutdown(
    bot: Bot,
    config: Config,
    database: Database,
    key_manager: ApiKeyManager,
    ai_client: AiClient,
) -> None:
    logger.info("Shutting down...")

    for admin_id in config.admin_ids:
        try:
            await bot.send_message(admin_id, "⚠️ Бот останавливается...")
        except Exception:
            pass

    await key_manager.stop_recovery_loop()
    await ai_client.close()
    await database.update_daily_stats()
    await database.close()
    logger.info("Shutdown complete.")


async def stats_updater(database: Database) -> None:
    while True:
        try:
            await asyncio.sleep(600)  # каждые 10 минут
            await database.update_daily_stats()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Stats updater error: %s", e)
            await asyncio.sleep(60)


async def main() -> None:
    config = load_config()
    setup_logging(config)

    logger.info("Initializing bot...")

    database = Database(config.db.path)
    key_manager = ApiKeyManager(config, database)
    ai_client = AiClient(config, key_manager)
    context_manager = ContextManager(config, database)

    bot = Bot(
        token=config.bot.token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()

    # Сохраняем зависимости в workflow_data для инъекции в хэндлеры
    dp.workflow_data.update({
        "config": config,
        "database": database,
        "key_manager": key_manager,
        "ai_client": ai_client,
        "context_manager": context_manager,
    })

    # Middleware
    dp.message.middleware(LoggingMiddleware())
    dp.message.middleware(UserRegistrationMiddleware(database))
    dp.message.middleware(ThrottleMiddleware(rate_limit=1.0))
    dp.callback_query.middleware(CallbackRegistrationMiddleware(database))

    # Роутеры (порядок важен: admin перед user, чтобы /ban /unban /broadcast не попали в AI)
    dp.include_router(admin.router)
    dp.include_router(callbacks.router)
    dp.include_router(user.router)

    # Фоновая задача обновления статистики
    stats_task: asyncio.Task | None = None

    @dp.startup()
    async def startup_handler() -> None:
        nonlocal stats_task
        await on_startup(bot, config, database, key_manager, ai_client)
        stats_task = asyncio.create_task(stats_updater(database))

    @dp.shutdown()
    async def shutdown_handler() -> None:
        if stats_task:
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
        await on_shutdown(bot, config, database, key_manager, ai_client)

    logger.info("Starting polling...")
    try:
        await dp.start_polling(
            bot,
            allowed_updates=["message", "callback_query"],
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by KeyboardInterrupt")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)