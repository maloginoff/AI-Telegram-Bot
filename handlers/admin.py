import logging
import os
import platform
import time
from datetime import datetime

import psutil
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ParseMode

from config import Config
from database import Database
from api_manager import ApiKeyManager

logger = logging.getLogger(__name__)
router = Router(name="admin")

BOT_START_TIME = time.monotonic()


def _is_admin(user_id: int, config: Config) -> bool:
    return user_id in config.admin_ids


def _admin_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats"),
            InlineKeyboardButton(text="🔑 API-ключи", callback_data="adm:keys"),
        ],
        [
            InlineKeyboardButton(text="👥 Пользователи", callback_data="adm:users"),
            InlineKeyboardButton(text="⚙️ Система", callback_data="adm:system"),
        ],
        [
            InlineKeyboardButton(text="📨 Рассылка", callback_data="adm:broadcast"),
        ],
    ])


@router.message(Command("admin"))
async def cmd_admin(message: Message, config: Config) -> None:
    if not _is_admin(message.from_user.id, config):
        await message.answer("⛔ Нет доступа.")
        return

    await message.answer(
        "🛠 **Админ-панель**\nВыберите раздел:",
        reply_markup=_admin_main_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )


# ──────────── Stats ────────────

async def _build_stats_text(db: Database) -> str:
    total_users = await db.get_total_users()
    total_messages = await db.get_total_messages()
    today_messages = await db.get_messages_today()
    avg_response = await db.get_avg_response_time()

    return (
        "📊 **Статистика бота**\n\n"
        f"👥 Всего пользователей: **{total_users}**\n"
        f"💬 Сообщений сегодня: **{today_messages}**\n"
        f"💬 Сообщений всего: **{total_messages}**\n"
        f"⏱ Среднее время ответа (сегодня): **{avg_response} мс**"
    )


@router.callback_query(F.data == "adm:stats")
async def cb_stats(callback: CallbackQuery, config: Config, database: Database) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    text = await _build_stats_text(database)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:stats")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="adm:main")],
    ])
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    await callback.answer()


# ──────────── API Keys ────────────

async def _build_keys_text(key_manager: ApiKeyManager) -> str:
    all_keys = await key_manager.get_all_keys_status()
    if not all_keys:
        return "🔑 **API-ключи**\n\nКлючи не найдены."

    lines = ["🔑 **API-ключи**\n"]
    current_provider = ""

    for k in all_keys:
        if k["provider"] != current_provider:
            current_provider = k["provider"]
            lines.append(f"\n**{current_provider.upper()}:**")

        status_emoji = {"active": "🟢", "exhausted": "🟡", "error": "🔴"}.get(
            k["status"], "⚪"
        )
        lines.append(
            f"{status_emoji} `{k['key_hash']}` — "
            f"{k['status']} | "
            f"Запросов: {k['total_requests']} | "
            f"Исчерпан: {k['exhausted_count']}x"
        )
        if k["last_used"]:
            lines.append(f"   Посл. использование: {k['last_used']}")

    return "\n".join(lines)


def _build_keys_keyboard(all_keys: list[dict]) -> InlineKeyboardMarkup:
    buttons = []
    for k in all_keys:
        key_hash = k["key_hash"]
        short = key_hash[:8]
        if k["status"] == "active":
            buttons.append([InlineKeyboardButton(
                text=f"⏸ Исчерпать {short}",
                callback_data=f"adm:key_exhaust:{key_hash}",
            )])
        elif k["status"] in ("exhausted", "error"):
            buttons.append([InlineKeyboardButton(
                text=f"▶️ Активировать {short}",
                callback_data=f"adm:key_activate:{key_hash}",
            )])

    buttons.append([InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:keys")])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="adm:main")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.callback_query(F.data == "adm:keys")
async def cb_keys(
    callback: CallbackQuery, config: Config, key_manager: ApiKeyManager
) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    text = await _build_keys_text(key_manager)
    all_keys = await key_manager.get_all_keys_status()
    keyboard = _build_keys_keyboard(all_keys)
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    await callback.answer()


@router.callback_query(F.data.startswith("adm:key_exhaust:"))
async def cb_key_exhaust(
    callback: CallbackQuery, config: Config, key_manager: ApiKeyManager
) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    key_hash = callback.data.split(":", 2)[2]
    all_keys = await key_manager.get_all_keys_status()
    target = next((k for k in all_keys if k["key_hash"] == key_hash), None)
    if target:
        await key_manager.mark_exhausted(key_hash, target["provider"])
        await callback.answer(f"Ключ {key_hash[:8]} помечен как exhausted", show_alert=True)
    else:
        await callback.answer("Ключ не найден", show_alert=True)

    # Обновляем список
    text = await _build_keys_text(key_manager)
    all_keys = await key_manager.get_all_keys_status()
    keyboard = _build_keys_keyboard(all_keys)
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)


@router.callback_query(F.data.startswith("adm:key_activate:"))
async def cb_key_activate(
    callback: CallbackQuery, config: Config, key_manager: ApiKeyManager
) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    key_hash = callback.data.split(":", 2)[2]
    all_keys = await key_manager.get_all_keys_status()
    target = next((k for k in all_keys if k["key_hash"] == key_hash), None)
    if target:
        await key_manager.mark_active(key_hash, target["provider"])
        await callback.answer(f"Ключ {key_hash[:8]} активирован", show_alert=True)
    else:
        await callback.answer("Ключ не найден", show_alert=True)

    text = await _build_keys_text(key_manager)
    all_keys = await key_manager.get_all_keys_status()
    keyboard = _build_keys_keyboard(all_keys)
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)


# ──────────── Users ────────────

async def _build_users_text(db: Database) -> str:
    top = await db.get_top_users(10)
    if not top:
        return "👥 **Пользователи**\n\nПока нет пользователей."

    lines = ["👥 **Топ-10 активных пользователей:**\n"]
    for i, u in enumerate(top, 1):
        name = u["username"] or u["first_name"] or "Unknown"
        lines.append(
            f"{i}. **{name}** (ID: `{u['user_id']}`)\n"
            f"   Сообщений: {u['total_messages']} | "
            f"Последняя активность: {u['last_active'] or 'н/д'}"
        )
    return "\n".join(lines)


@router.callback_query(F.data == "adm:users")
async def cb_users(callback: CallbackQuery, config: Config, database: Database) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    text = await _build_users_text(database)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🚫 Забанить", callback_data="adm:ban_prompt"),
            InlineKeyboardButton(text="✅ Разбанить", callback_data="adm:unban_prompt"),
        ],
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:users")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="adm:main")],
    ])
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    await callback.answer()


@router.callback_query(F.data == "adm:ban_prompt")
async def cb_ban_prompt(callback: CallbackQuery, config: Config) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    await callback.message.answer(
        "Отправьте команду:\n`/ban <user_id>`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer()


@router.callback_query(F.data == "adm:unban_prompt")
async def cb_unban_prompt(callback: CallbackQuery, config: Config) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return
    await callback.message.answer(
        "Отправьте команду:\n`/unban <user_id>`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer()


@router.message(Command("ban"))
async def cmd_ban(message: Message, config: Config, database: Database) -> None:
    if not _is_admin(message.from_user.id, config):
        return

    parts = message.text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Использование: `/ban <user_id>`", parse_mode=ParseMode.MARKDOWN)
        return

    target_id = int(parts[1])
    if target_id in config.admin_ids:
        await message.answer("❌ Нельзя забанить администратора.")
        return

    success = await database.set_ban(target_id, True)
    if success:
        await message.answer(f"🚫 Пользователь `{target_id}` забанен.", parse_mode=ParseMode.MARKDOWN)
    else:
        await message.answer(f"❌ Пользователь `{target_id}` не найден.", parse_mode=ParseMode.MARKDOWN)


@router.message(Command("unban"))
async def cmd_unban(message: Message, config: Config, database: Database) -> None:
    if not _is_admin(message.from_user.id, config):
        return

    parts = message.text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Использование: `/unban <user_id>`", parse_mode=ParseMode.MARKDOWN)
        return

    target_id = int(parts[1])
    success = await database.set_ban(target_id, False)
    if success:
        await message.answer(f"✅ Пользователь `{target_id}` разбанен.", parse_mode=ParseMode.MARKDOWN)
    else:
        await message.answer(f"❌ Пользователь `{target_id}` не найден.", parse_mode=ParseMode.MARKDOWN)


# ──────────── System ────────────

def _build_system_text() -> str:
    uptime_seconds = time.monotonic() - BOT_START_TIME
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}ч {minutes}м {seconds}с"

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)

    total_mem = psutil.virtual_memory()
    sys_mem_used = total_mem.percent

    return (
        "⚙️ **Информация о системе**\n\n"
        f"⏱ Аптайм: **{uptime_str}**\n"
        f"🧠 RAM бота: **{mem_mb:.1f} MB**\n"
        f"💻 CPU бота: **{cpu_percent:.1f}%**\n"
        f"📦 RAM системы: **{sys_mem_used:.1f}%**\n"
        f"🐍 Python: **{platform.python_version()}**\n"
        f"💿 ОС: **{platform.system()} {platform.release()}**"
    )


@router.callback_query(F.data == "adm:system")
async def cb_system(callback: CallbackQuery, config: Config) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    text = _build_system_text()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:system")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="adm:main")],
    ])
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    await callback.answer()


# ──────────── Broadcast ────────────

@router.callback_query(F.data == "adm:broadcast")
async def cb_broadcast_prompt(callback: CallbackQuery, config: Config) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    await callback.message.answer(
        "📨 Отправьте команду:\n`/broadcast <текст сообщения>`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer()


@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message, config: Config, database: Database) -> None:
    if not _is_admin(message.from_user.id, config):
        return

    text = message.text.partition(" ")[2].strip()
    if not text:
        await message.answer(
            "Использование: `/broadcast <текст>`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    user_ids = await database.get_all_user_ids()
    if not user_ids:
        await message.answer("Нет пользователей для рассылки.")
        return

    status_msg = await message.answer(f"📨 Рассылка {len(user_ids)} пользователям...")

    sent = 0
    failed = 0
    for uid in user_ids:
        try:
            await message.bot.send_message(uid, text)
            sent += 1
        except Exception:
            failed += 1

        if (sent + failed) % 25 == 0:
            try:
                await status_msg.edit_text(
                    f"📨 Рассылка... {sent + failed}/{len(user_ids)}"
                )
            except Exception:
                pass
            # Telegram rate limit: ~30 msg/sec
            await asyncio.sleep(1)

    await status_msg.edit_text(
        f"✅ Рассылка завершена.\n"
        f"Отправлено: {sent}\n"
        f"Ошибок: {failed}"
    )


# ──────────── Back to main ────────────

@router.callback_query(F.data == "adm:main")
async def cb_main(callback: CallbackQuery, config: Config) -> None:
    if not _is_admin(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    await callback.message.edit_text(
        "🛠 **Админ-панель**\nВыберите раздел:",
        reply_markup=_admin_main_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer()