import logging

from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.enums import ParseMode

from config import Config
from context_manager import ContextManager
from api_manager import ApiKeyManager

logger = logging.getLogger(__name__)
router = Router(name="callbacks")


# Тот же маппинг что и в user.py
MODEL_MAP = {
    "or1": ("openrouter", "google/gemini-2.0-flash-exp:free"),
    "or2": ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
    "or3": ("openrouter", "mistralai/mistral-small-24b-instruct-2501:free"),
    "or4": ("openrouter", "qwen/qwen2.5-vl-72b-instruct:free"),
    "gm1": ("gemini", "gemini-2.0-flash"),
    "gm2": ("gemini", "gemini-1.5-flash"),
    "gm3": ("gemini", "gemini-1.5-pro"),
}

MODEL_NAMES = {
    "or1": "Gemini 2.0 Flash (free)",
    "or2": "Llama 3.3 70B (free)",
    "or3": "Mistral Small 24B (free)",
    "or4": "Qwen 2.5 VL 72B (free)",
    "gm1": "Gemini 2.0 Flash",
    "gm2": "Gemini 1.5 Flash",
    "gm3": "Gemini 1.5 Pro",
}


@router.callback_query(F.data.startswith("sm:"))
async def cb_set_model(
    callback: CallbackQuery,
    config: Config,
    context_manager: ContextManager,
    key_manager: ApiKeyManager,
) -> None:
    short_id = callback.data.split(":", 1)[1]

    if short_id not in MODEL_MAP:
        await callback.answer("❌ Модель не найдена.", show_alert=True)
        return

    provider, model_id = MODEL_MAP[short_id]
    display_name = MODEL_NAMES.get(short_id, model_id)

    has_keys = await key_manager.has_active_keys(provider)
    if not has_keys:
        all_keys = await key_manager.get_all_keys_status(provider)
        if not all_keys:
            await callback.answer(
                f"❌ Нет API-ключей для {provider}.",
                show_alert=True,
            )
            return
        await callback.answer(
            f"⚠️ Все ключи {provider} временно недоступны.",
            show_alert=True,
        )
        return

    await context_manager.set_user_model(callback.from_user.id, provider, model_id)

    await callback.message.edit_text(
        f"✅ Модель изменена!\n\n"
        f"🤖 **{display_name}**\n"
        f"Провайдер: `{provider}`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer("Модель выбрана!")


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await callback.answer()