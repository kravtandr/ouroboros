"""
Video Generation Bot — Telegram бот для генерации видео
Часть AnimateYou, работает с той же БД и воркерами.
"""
import asyncio
import io
import os
import uuid
import httpx
from datetime import datetime, timezone
from typing import Optional

from telegram import (
    Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.models import (
    User, Generation, GenerationStatus, GenerationType,
    Transaction, TransactionType, TransactionStatus, ModerationStrategy
)
from app.config import get_settings
from app.services.unified_storage import get_unified_storage
from app.services import GenerationService
from app.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("video_bot")

settings = get_settings()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is required")

ADMIN_IDS_ENV = os.getenv("ADMIN_IDS", "398548347")
try:
    ADMIN_IDS = [int(x.strip()) for x in ADMIN_IDS_ENV.split(",") if x.strip()]
except ValueError:
    ADMIN_IDS = [398548347]

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")

engine = create_engine(DATABASE_URL)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

storage = get_unified_storage()
generation_service = GenerationService()

# ──────────────────────────────────────────────────────────────
# In-memory state per user:
# {
#   user_id: {
#     "step": "idle" | "await_prompt" | "await_model" | "await_aspect",
#     "file_url": str,          # MinIO URL of uploaded photo
#     "prompt": str | None,
#     "model": str | None,
#   }
# }
# ──────────────────────────────────────────────────────────────
user_states: dict[int, dict] = {}

MODELS = {
    "standard": {
        "id": "gptunnel_advanced",
        "label": "🎬 Стандарт",
        "cost": 30,
        "description": "Стандартное качество, быстрая генерация",
    },
    "premium": {
        "id": "premium",
        "label": "⭐ Премиум",
        "cost": 50,
        "description": "Высокое качество, SeeDance-1-Pro",
    },
}

ASPECT_OPTIONS = [
    ("1:1", "⬛ 1:1 Квадрат"),
    ("9:16", "📱 9:16 Вертикаль"),
    ("16:9", "🖥 16:9 Горизонталь"),
]


# ──────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────

def get_or_create_user(telegram_user) -> User:
    db = SessionFactory()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_user.id).first()
        if not user:
            user = User(
                telegram_id=telegram_user.id,
                username=telegram_user.username,
                first_name=telegram_user.first_name or "Пользователь",
                last_name=telegram_user.last_name,
                balance=settings.new_user_starting_balance,
                free_generations_remaining=0,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user registered: {telegram_user.id}")
        return user
    finally:
        db.close()


def get_user(telegram_id: int) -> Optional[User]:
    db = SessionFactory()
    try:
        return db.query(User).filter(User.telegram_id == telegram_id).first()
    finally:
        db.close()


def add_balance(target_telegram_id: int, amount: int, admin_id: int) -> tuple[bool, str]:
    db = SessionFactory()
    try:
        user = db.query(User).filter(User.telegram_id == target_telegram_id).with_for_update().first()
        if not user:
            return False, "Пользователь не найден"
        balance_before = user.balance
        user.balance += amount
        tx = Transaction(
            user_id=user.id,
            type=TransactionType.DEPOSIT.value,
            status=TransactionStatus.COMPLETED.value,
            amount=amount,
            balance_before=balance_before,
            balance_after=user.balance,
            description=f"Админское пополнение от {admin_id}",
            completed_at=datetime.now(timezone.utc),
        )
        db.add(tx)
        db.commit()
        return True, f"Баланс: {balance_before} → {user.balance}"
    except Exception as e:
        db.rollback()
        return False, str(e)
    finally:
        db.close()


async def upload_telegram_photo(bot: Bot, photo) -> str:
    """Download photo from Telegram and upload to MinIO."""
    tg_file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    buf.seek(0)
    file_bytes = buf.read()
    filename = f"tgbot_{photo.file_unique_id}.jpg"
    url = await storage.upload_file(
        file_data=file_bytes,
        filename=filename,
        content_type="image/jpeg",
        bucket=settings.minio_bucket_uploads,
    )
    return url


async def start_generation(
    bot: Bot,
    chat_id: int,
    telegram_user,
    file_url: str,
    prompt: Optional[str],
    model_key: str,
    aspect_ratio: str,
) -> None:
    """Create generation record and dispatch to worker."""
    db = SessionFactory()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_user.id).with_for_update().first()
        if not user:
            await bot.send_message(chat_id, "❌ Пользователь не найден.")
            return

        model_cfg = MODELS[model_key]
        model_id = model_cfg["id"]
        cost = model_cfg["cost"]

        if user.balance < cost:
            await bot.send_message(
                chat_id,
                f"❌ Недостаточно токенов.\n"
                f"Нужно: {cost} токенов\n"
                f"У вас: {user.balance} токенов\n\n"
                f"Пополните баланс в WebApp AnimateYou.",
            )
            return

        # Deduct balance
        balance_before = user.balance
        user.balance -= cost
        tx = Transaction(
            user_id=user.id,
            type=TransactionType.WITHDRAWAL.value,
            status=TransactionStatus.COMPLETED.value,
            amount=-cost,
            balance_before=balance_before,
            balance_after=user.balance,
            description=f"Генерация видео ({model_id})",
            completed_at=datetime.now(timezone.utc),
        )
        db.add(tx)

        parameters = {
            "aspect_ratio": aspect_ratio,
            "duration": "5s",
        }

        generation = Generation(
            user_id=user.id,
            status=GenerationStatus.PENDING,
            type=GenerationType.PREMIUM,
            input_file_url=file_url,
            prompt=prompt or "Animate this image with smooth motion",
            original_prompt=prompt or "",
            model=model_id,
            parameters=parameters,
            cost=cost,
            moderation_strategy=ModerationStrategy.ALLOW_EXPLICIT,
            moderation_passed=True,
            strict_image_moderation_failed=False,
        )
        db.add(generation)
        db.commit()
        db.refresh(generation)

        generation_id = generation.id
        logger.info(f"Created generation {generation_id} for user {telegram_user.id}")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create generation: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Ошибка при создании генерации: {e}")
        return
    finally:
        db.close()

    await bot.send_message(
        chat_id,
        f"⏳ Генерация видео запущена!\n"
        f"Модель: {model_cfg['label']}\n"
        f"Соотношение: {aspect_ratio}\n\n"
        f"Обычно занимает 1–3 минуты. Ожидайте...",
    )

    # Dispatch to worker
    task_data = {
        "generation_id": str(generation_id),
        "user_id": user.id,
        "input_file_url": file_url,
        "generation_type": GenerationType.PREMIUM.value,
        "model": model_id,
        "parameters": parameters,
        "prompt": prompt or "Animate this image with smooth motion",
        "original_prompt": prompt or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    async def _run():
        from workers.generation_worker import process_generation
        await process_generation(task_data)

    asyncio.create_task(_run())

    # Poll for completion
    asyncio.create_task(
        poll_and_notify(bot, chat_id, generation_id, telegram_user)
    )


async def poll_and_notify(bot: Bot, chat_id: int, generation_id: uuid.UUID, telegram_user) -> None:
    """Poll generation status and send result when done."""
    for attempt in range(72):  # 6 minutes max
        await asyncio.sleep(5)
        db = SessionFactory()
        try:
            gen = db.query(Generation).filter(Generation.id == generation_id).first()
            if not gen:
                logger.warning(f"Generation {generation_id} not found while polling")
                return

            if gen.status == GenerationStatus.COMPLETED and gen.output_file_url:
                output_url = gen.output_file_url
                public_url = storage.get_public_url(output_url)
                logger.info(f"Generation {generation_id} completed: {public_url}")

                # Download video and send
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.get(public_url)
                        resp.raise_for_status()
                        video_bytes = resp.content

                    model_label = next(
                        (m["label"] for m in MODELS.values() if m["id"] == gen.model),
                        gen.model,
                    )
                    caption = (
                        f"✅ Видео готово!\n"
                        f"Модель: {model_label}\n"
                        f"Промпт: {gen.prompt[:100] if gen.prompt else '—'}"
                    )
                    await bot.send_video(
                        chat_id=chat_id,
                        video=InputFile(io.BytesIO(video_bytes), filename="animation.mp4"),
                        caption=caption,
                        supports_streaming=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to send video: {e}", exc_info=True)
                    await bot.send_message(
                        chat_id,
                        f"✅ Видео готово! Смотри по ссылке:\n{public_url}",
                    )
                return

            elif gen.status == GenerationStatus.FAILED:
                await bot.send_message(
                    chat_id,
                    "❌ Генерация не удалась. Токены возвращены на баланс.",
                )
                return

        except Exception as e:
            logger.error(f"Polling error for {generation_id}: {e}")
        finally:
            db.close()

    await bot.send_message(chat_id, "⏰ Время ожидания истекло. Попробуйте ещё раз.")


# ──────────────────────────────────────────────────────────────
# Keyboards
# ──────────────────────────────────────────────────────────────

def photo_received_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✏️ Добавить описание", callback_data="add_prompt"),
            InlineKeyboardButton("⚡ Быстрая генерация", callback_data="quick_gen"),
        ],
    ])


def model_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for key, cfg in MODELS.items():
        rows.append([
            InlineKeyboardButton(
                f"{cfg['label']} — {cfg['cost']} токенов",
                callback_data=f"model:{key}",
            )
        ])
    rows.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_photo")])
    return InlineKeyboardMarkup(rows)


def aspect_keyboard(model_key: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(label, callback_data=f"aspect:{model_key}:{ratio}")]
        for ratio, label in ASPECT_OPTIONS
    ]
    rows.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_model")])
    return InlineKeyboardMarkup(rows)


# ──────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = update.effective_user
    get_or_create_user(tg_user)
    user_states[tg_user.id] = {"step": "idle"}

    await update.message.reply_text(
        f"👋 Привет, {tg_user.first_name}!\n\n"
        "🎬 *AnimateYou Video Bot* — генерирую видео из твоих фото прямо в Telegram.\n\n"
        "📸 Отправь фото, чтобы начать.\n\n"
        "Команды:\n"
        "/balance — баланс токенов\n"
        "/history — последние генерации\n"
        "/help — помощь",
        parse_mode="Markdown",
    )


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = update.effective_user
    user = get_or_create_user(tg_user)
    await update.message.reply_text(
        f"💰 Ваш баланс: *{user.balance} токенов*\n\n"
        f"Стоимость генерации:\n"
        f"• Стандарт — 30 токенов\n"
        f"• Премиум — 50 токенов",
        parse_mode="Markdown",
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = update.effective_user
    db = SessionFactory()
    try:
        user = db.query(User).filter(User.telegram_id == tg_user.id).first()
        if not user:
            await update.message.reply_text("Вы ещё не делали генераций.")
            return
        gens = (
            db.query(Generation)
            .filter(Generation.user_id == user.id)
            .order_by(Generation.created_at.desc())
            .limit(5)
            .all()
        )
        if not gens:
            await update.message.reply_text("История генераций пуста.")
            return
        lines = ["📋 *Последние генерации:*\n"]
        for g in gens:
            status_icon = {
                GenerationStatus.COMPLETED: "✅",
                GenerationStatus.FAILED: "❌",
                GenerationStatus.PROCESSING: "⏳",
                GenerationStatus.PENDING: "🕐",
            }.get(g.status, "❓")
            dt = g.created_at.strftime("%d.%m %H:%M") if g.created_at else "—"
            lines.append(f"{status_icon} {dt} | {g.model} | {g.cost} токенов")
            if g.prompt:
                lines.append(f"   _{g.prompt[:60]}..._" if len(g.prompt) > 60 else f"   _{g.prompt}_")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    finally:
        db.close()


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Как пользоваться:*\n\n"
        "1. Отправь фото\n"
        "2. Выбери: добавить описание или быструю генерацию\n"
        "3. Выбери модель и соотношение сторон\n"
        "4. Получи видео через 1–3 минуты!\n\n"
        "*Модели:*\n"
        "• 🎬 Стандарт (30 токенов) — gptunnel_advanced\n"
        "• ⭐ Премиум (50 токенов) — SeeDance-1-Pro\n\n"
        "*Команды:*\n"
        "/balance — ваш баланс\n"
        "/history — история генераций\n"
        "/start — начать заново",
        parse_mode="Markdown",
    )


async def cmd_adm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    if len(context.args) != 2:
        await update.message.reply_text("Использование: /adm <telegram_id> <amount>")
        return
    try:
        target_id = int(context.args[0])
        amount = int(context.args[1])
    except ValueError:
        await update.message.reply_text("Неверный формат аргументов")
        return
    ok, msg = add_balance(target_id, amount, update.effective_user.id)
    if ok:
        await update.message.reply_text(f"✅ Готово. {msg}")
        try:
            await context.bot.send_message(
                target_id,
                f"💰 Вам начислено {amount} токенов администратором.\n{msg}",
            )
        except Exception:
            pass
    else:
        await update.message.reply_text(f"❌ Ошибка: {msg}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    db = SessionFactory()
    try:
        from sqlalchemy import cast, Date
        total_users = db.query(func.count(User.id)).scalar()
        today = datetime.now(timezone.utc).date()
        gens_today = (
            db.query(func.count(Generation.id))
            .filter(func.date(Generation.created_at) == today)
            .scalar()
        )
        completed_today = (
            db.query(func.count(Generation.id))
            .filter(
                func.date(Generation.created_at) == today,
                Generation.status == GenerationStatus.COMPLETED,
            )
            .scalar()
        )
        await update.message.reply_text(
            f"📊 *Статистика:*\n\n"
            f"👥 Всего пользователей: {total_users}\n"
            f"🎬 Генераций сегодня: {gens_today}\n"
            f"✅ Завершено сегодня: {completed_today}",
            parse_mode="Markdown",
        )
    finally:
        db.close()


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = update.effective_user
    get_or_create_user(tg_user)

    # Take the largest photo
    photo = update.message.photo[-1]

    msg = await update.message.reply_text("📤 Загружаю фото...")
    try:
        file_url = await upload_telegram_photo(context.bot, photo)
    except Exception as e:
        logger.error(f"Failed to upload photo: {e}", exc_info=True)
        await msg.edit_text("❌ Не удалось загрузить фото. Попробуйте ещё раз.")
        return

    user_states[tg_user.id] = {
        "step": "await_action",
        "file_url": file_url,
        "prompt": None,
        "model": None,
    }

    await msg.edit_text(
        "📸 Фото загружено! Что дальше?",
        reply_markup=photo_received_keyboard(),
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = update.effective_user
    state = user_states.get(tg_user.id, {})

    if state.get("step") == "await_prompt":
        prompt = update.message.text.strip()
        state["prompt"] = prompt
        state["step"] = "await_model"

        await update.message.reply_text(
            f"✅ Описание сохранено:\n_{prompt}_\n\n"
            "Выберите модель генерации:",
            parse_mode="Markdown",
            reply_markup=model_keyboard(),
        )
    else:
        await update.message.reply_text(
            "📸 Отправьте фото, чтобы начать генерацию."
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    tg_user = query.from_user
    state = user_states.get(tg_user.id, {})
    data = query.data

    if data == "add_prompt":
        state["step"] = "await_prompt"
        await query.edit_message_text(
            "✏️ Опишите желаемое движение или стиль анимации:\n\n"
            "_Пример: Плавное покачивание волос, мягкий ветер_",
            parse_mode="Markdown",
        )

    elif data == "quick_gen":
        state["prompt"] = None
        state["step"] = "await_model"
        await query.edit_message_text(
            "⚡ Быстрая генерация — промпт создастся автоматически.\n\n"
            "Выберите модель:",
            reply_markup=model_keyboard(),
        )

    elif data.startswith("model:"):
        model_key = data.split(":", 1)[1]
        if model_key not in MODELS:
            await query.edit_message_text("❌ Неизвестная модель")
            return
        state["model"] = model_key
        state["step"] = "await_aspect"
        cfg = MODELS[model_key]
        await query.edit_message_text(
            f"Модель: *{cfg['label']}* — {cfg['cost']} токенов\n\n"
            "Выберите соотношение сторон:",
            parse_mode="Markdown",
            reply_markup=aspect_keyboard(model_key),
        )

    elif data.startswith("aspect:"):
        _, model_key, aspect_ratio = data.split(":", 2)
        file_url = state.get("file_url")
        prompt = state.get("prompt")

        if not file_url:
            await query.edit_message_text("❌ Фото не найдено. Отправьте фото заново.")
            return

        user_states[tg_user.id] = {"step": "idle"}
        await query.edit_message_text("⏳ Запускаю генерацию...")

        await start_generation(
            bot=context.bot,
            chat_id=query.message.chat_id,
            telegram_user=tg_user,
            file_url=file_url,
            prompt=prompt,
            model_key=model_key,
            aspect_ratio=aspect_ratio,
        )

    elif data == "back_to_photo":
        state["step"] = "await_action"
        state["model"] = None
        await query.edit_message_text(
            "📸 Что дальше?",
            reply_markup=photo_received_keyboard(),
        )

    elif data == "back_to_model":
        state["step"] = "await_model"
        await query.edit_message_text(
            "Выберите модель генерации:",
            reply_markup=model_keyboard(),
        )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("adm", cmd_adm))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("🎬 VideoBot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
