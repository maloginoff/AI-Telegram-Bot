import asyncio
import logging
import time

from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ParseMode

from config import Config
from ai_client import AiClient, AiError, AllKeysExhaustedError
from context_manager import ContextManager

logger = logging.getLogger(__name__)
router = Router(name="user")

WELCOME_TEXT = """👋 **Привет, {name}!**

Я — AI-ассистент. Просто напиши мне сообщение, и я отвечу.

🔹 /models — выбрать модель AI
🔹 /clear — очистить историю диалога
🔹 /model — текущая модель
🔹 /help — все команды"""

HELP_TEXT = """📖 **Список команд:**

🔹 /start — приветствие
🔹 /help — это сообщение
🔹 /model — текущая модель
🔹 /models — выбрать модель
🔹 /clear — очистить контекст

Просто отправь текстовое сообщение — я отвечу с помощью AI.
Бот помнит последние 15 сообщений диалога."""

# Короткие ID для callback_data (лимит Telegram — 64 байта)
MODEL_MAP = {
    "or1": ("openrouter", "google/gemini-2.0-flash-exp:free"),
    "or2": ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
    "or3": ("openrouter", "mistralai/mistral-small-24b-instruct-2501:free"),
    "or4": ("openrouter", "qwen/qwen2.5-vl-72b-instruct:free"),
    "gm1": ("gemini", "gemini-2.0-flash"),
    "gm2": ("gemini", "gemini-1.5-flash"),
    "gm3": ("gemini", "gemini-1.5-pro"),
}

# Обратный маппинг для отображения
MODEL_NAMES = {
    "or1": "Gemini 2.0 Flash (free)",
    "or2": "Llama 3.3 70B (free)",
    "or3": "Mistral Small 24B (free)",
    "or4": "Qwen 2.5 VL 72B (free)",
    "gm1": "Gemini 2.0 Flash",
    "gm2": "Gemini 1.5 Flash",
    "gm3": "Gemini 1.5 Pro",
}


@router.message(CommandStart())
async def cmd_start(message: Message, config: Config) -> None:
    name = message.from_user.first_name or "друг"
    await message.answer(
        WELCOME_TEXT.format(name=name),
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


@router.message(Command("clear"))
async def cmd_clear(message: Message, context_manager: ContextManager) -> None:
    count = await context_manager.clear(message.from_user.id)
    await message.answer(f"🗑 Контекст очищен ({count} сообщений удалено).")


@router.message(Command("model"))
async def cmd_model(message: Message, context_manager: ContextManager, config: Config) -> None:
    provider, model = await context_manager.get_user_model(message.from_user.id)
    model_info = config.get_model_info(model)
    display_name = model_info.name if model_info else model
    await message.answer(
        f"🤖 **Текущая модель:**\n"
        f"Провайдер: `{provider}`\n"
        f"Модель: `{display_name}`",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("models"))
async def cmd_models(message: Message, config: Config) -> None:
    keyboard = _build_models_keyboard(config)
    await message.answer(
        "🔧 **Выберите модель:**",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )


def _build_models_keyboard(config: Config) -> InlineKeyboardMarkup:
    buttons = []

    # OpenRouter
    or_keys = [k for k in MODEL_MAP if k.startswith("or")]
    if config.api.openrouter_keys:
        buttons.append([InlineKeyboardButton(
            text="── OpenRouter ──", callback_data="noop"
        )])
        for key in or_keys:
            buttons.append([InlineKeyboardButton(
                text=f"🟢 {MODEL_NAMES[key]}", callback_data=f"sm:{key}"
            )])

    # Gemini
    gm_keys = [k for k in MODEL_MAP if k.startswith("gm")]
    if config.api.gemini_keys:
        buttons.append([InlineKeyboardButton(
            text="── Google Gemini ──", callback_data="noop"
        )])
        for key in gm_keys:
            buttons.append([InlineKeyboardButton(
                text=f"🔵 {MODEL_NAMES[key]}", callback_data=f"sm:{key}"
            )])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_message(
    message: Message,
    config: Config,
    ai_client: AiClient,
    context_manager: ContextManager,
) -> None:
    user_id = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    provider, model = await context_manager.get_user_model(user_id)

    await context_manager.add_user_message(user_id, user_text)

    messages = await context_manager.get_messages_for_request(user_id)

    thinking_msg = await message.answer("💭 Думаю...")

    start_time = time.monotonic()
    full_response = ""
    last_edit_time = 0.0

    try:
        async for chunk in ai_client.stream_response(messages, model, provider):
            full_response += chunk

            now = time.monotonic()
            if now - last_edit_time >= 1.5 and full_response.strip():
                try:
                    display = full_response
                    if len(display) > 4000:
                        display = display[:4000] + "…"
                    await thinking_msg.edit_text(display + " ▌")
                    last_edit_time = now
                except Exception:
                    pass

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if full_response.strip():
            display = full_response
            if len(display) > 4000:
                display = display[:4000] + "…"
            try:
                await thinking_msg.edit_text(display, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                try:
                    await thinking_msg.edit_text(display)
                except Exception:
                    pass

            await context_manager.add_assistant_message(
                user_id, full_response, model, elapsed_ms
            )
        else:
            await thinking_msg.edit_text("😶 Получен пустой ответ. Попробуйте ещё раз или смените модель /models")

    except AllKeysExhaustedError as e:
        await thinking_msg.edit_text(f"⚠️ {e}")

    except AiError as e:
        await thinking_msg.edit_text(f"❌ Ошибка AI: {e}")

    except asyncio.TimeoutError:
        await thinking_msg.edit_text(
            "⏱ Превышено время ожидания. Попробуйте позже или смените модель /models"
        )

    except Exception as e:
        logger.exception("Unexpected error in handle_message")
        await thinking_msg.edit_text(
            "💥 Произошла непредвиденная ошибка. Попробуйте позже."
        )