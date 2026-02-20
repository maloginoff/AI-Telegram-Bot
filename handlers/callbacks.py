import logging

from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.enums import ParseMode

from config import Config
from context_manager import ContextManager
from api_manager import ApiKeyManager

logger = logging.getLogger(__name__)
router = Router(name="callbacks")


@router.callback_query(F.data.startswith("setmodel:"))
async def cb_set_model(
    callback: CallbackQuery,
    config: Config,
    context_manager: ContextManager,
    key_manager: ApiKeyManager,
) -> None:
    parts = callback.data.split(":", 2)
    if len(parts) < 3:
        await callback.answer("❌ Некорректные данные.", show_alert=True)
        return

    provider = parts[1]
    model_id = parts[2]

    model_info = config.get_model_info(model_id)
    if not model_info:
        await callback.answer("❌ Модель не найдена.", show_alert=True)
        return

    has_keys = await key_manager.has_active_keys(provider)
    if not has_keys:
        all_keys_for_provider = await key_manager.get_all_keys_status(provider)
        if not all_keys_for_provider:
            await callback.answer(
                f"❌ Нет API-ключей для {provider}. Добавьте ключи в .env",
                show_alert=True,
            )
            return
        await callback.answer(
            f"⚠️ Все ключи {provider} временно недоступны. Попробуйте позже.",
            show_alert=True,
        )
        return

    await context_manager.set_user_model(callback.from_user.id, provider, model_id)

    await callback.message.edit_text(
        f"✅ Модель изменена!\n\n"
        f"🤖 **{model_info.name}**\n"
        f"Провайдер: `{provider}`\n"
        f"ID: `{model_id}`",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.answer("Модель выбрана!")


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await callback.answer()