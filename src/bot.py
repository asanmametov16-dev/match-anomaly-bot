"""Telegram-бот: команды для управления и просмотра состояния.

Запускается как фоновая задача. Принимает команды:
  /start, /help — список команд
  /stats        — общая статистика по аномалиям
  /recent [N]   — последние N аномалий (по умолчанию 5)
  /thresholds   — текущие пороги детекторов
  /elo TEAM     — текущий Elo-рейтинг команды
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from html import escape

from sqlalchemy import desc, func, select
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          filters)

from .config import settings
from .db import Anomaly, AnomalyOutcome, OddsSnapshot, SessionLocal, TeamRating
from .elo import _normalize as normalize_team

log = logging.getLogger(__name__)


def _is_authorized(update: Update) -> bool:
    """Принимаем команды только от owner-чата."""
    if update.effective_chat is None:
        return False
    return str(update.effective_chat.id) == str(settings.telegram_chat_id)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 Anomaly bot активен.\n\n"
        "Команды:\n"
        "/stats — общая статистика\n"
        "/accuracy — точность детекторов\n"
        "/recent [N] — последние N аномалий\n"
        "/thresholds — текущие пороги\n"
        "/elo <команда> — Elo-рейтинг\n"
        "/help — эта справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    with SessionLocal() as session:
        total = session.scalar(select(func.count(Anomaly.id))) or 0
        last_24h = session.scalar(
            select(func.count(Anomaly.id))
            .where(Anomaly.detected_at >= datetime.utcnow() - timedelta(hours=24))
        ) or 0
        snapshots = session.scalar(select(func.count(OddsSnapshot.id))) or 0
        teams = session.scalar(select(func.count(TeamRating.team))) or 0

        by_detector = session.execute(
            select(Anomaly.detector, func.count(Anomaly.id))
            .group_by(Anomaly.detector)
        ).all()

    lines = [
        "📊 <b>Статистика</b>",
        f"Всего аномалий: <b>{total}</b>",
        f"За последние 24ч: <b>{last_24h}</b>",
        f"Снимков коэффициентов: {snapshots}",
        f"Команд в Elo-таблице: {teams}",
        "",
        "<b>По детекторам:</b>",
    ]
    if by_detector:
        for detector, count in sorted(by_detector, key=lambda x: -x[1]):
            lines.append(f"  • {detector}: {count}")
    else:
        lines.append("  (пока нет срабатываний)")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    with SessionLocal() as session:
        rows = session.execute(
            select(
                AnomalyOutcome.detector,
                func.count(AnomalyOutcome.id),
                func.sum(AnomalyOutcome.confirmed),
            )
            .group_by(AnomalyOutcome.detector)
        ).all()
        total_matches = session.scalar(
            select(func.count()).select_from(
                select(AnomalyOutcome.result_key).distinct().subquery()
            )
        ) or 0

    if not rows:
        await update.message.reply_text(
            "Данных пока нет — результаты матчей ещё не проверялись.\n"
            "Проверка запускается каждые 4 часа (только топ-лиги)."
        )
        return

    lines = ["📈 <b>Точность детекторов</b>", f"Матчей с результатом: {total_matches}", ""]

    directional = []
    non_directional = []
    for detector, total, confirmed_sum in sorted(rows, key=lambda r: -r[1]):
        yes = int(confirmed_sum or 0)
        no = total - yes
        # confirmed=NULL не суммируется — это ненаправленные срабатывания
        directional_total = yes + no
        if directional_total == 0:
            non_directional.append(detector)
        else:
            pct = yes / directional_total * 100
            bar = "🟢" if pct >= 60 else ("🟡" if pct >= 40 else "🔴")
            directional.append(
                f"{bar} <b>{escape(detector)}</b>: "
                f"{yes}/{directional_total} = {pct:.0f}%"
            )

    if directional:
        lines += directional
    if non_directional:
        lines.append("")
        lines.append("<i>Без направления (только счёт): "
                     + ", ".join(non_directional) + "</i>")

    lines += [
        "",
        "<i>🟢 ≥60%  🟡 40–59%  🔴 &lt;40%</i>",
        "<i>Подтверждение = рынок двигался в сторону победителя.</i>",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    n = 5
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 20))
        except ValueError:
            pass

    with SessionLocal() as session:
        rows = session.execute(
            select(Anomaly).order_by(desc(Anomaly.detected_at)).limit(n)
        ).scalars().all()

    if not rows:
        await update.message.reply_text("Аномалий пока нет.")
        return

    lines = [f"📋 <b>Последние {len(rows)} аномалий:</b>", ""]
    for a in rows:
        when = a.detected_at.strftime("%m-%d %H:%M")
        lines.append(
            f"<b>{escape(a.home_team)} — {escape(a.away_team)}</b>\n"
            f"  {when} · {a.detector} · severity={a.severity:.2f}\n"
            f"  <i>{escape(a.details or '')}</i>"
        )
    await update.message.reply_text("\n\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_thresholds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    text = (
        "⚙️ <b>Текущие пороги детекторов</b>\n\n"
        f"spread (1X2 между букмекерами): <b>{settings.spread_threshold*100:.1f}%</b>\n"
        f"drift (движение от открытия): <b>{settings.drift_threshold*100:.1f}%</b>\n"
        f"model_gap (рынок vs Elo): <b>{settings.model_gap_threshold*100:.1f}%</b>\n"
        f"sync_move: <b>{settings.sync_move_threshold*100:.1f}%</b>, "
        f"мин. контор: <b>{settings.sync_min_bookmakers}</b>\n"
        f"exotic_spread (тоталы/форы): <b>{settings.exotic_spread_threshold*100:.1f}%</b>\n\n"
        "<i>Меняются через .env, требуют рестарт.</i>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_elo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Используй: /elo <название команды>")
        return
    name = " ".join(context.args)
    key = normalize_team(name)
    with SessionLocal() as session:
        row = session.get(TeamRating, key)
    if row is None:
        await update.message.reply_text(
            f"Команда '{escape(name)}' не найдена. "
            f"Возможно, ещё не было её матчей с момента запуска."
        )
        return
    await update.message.reply_text(
        f"<b>{escape(name)}</b>\n"
        f"Elo: {row.rating:.1f}\n"
        f"Игр учтено: {row.games_played}",
        parse_mode=ParseMode.HTML,
    )


def build_application() -> Application:
    """Создаёт приложение бота с зарегистрированными хендлерами."""
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("accuracy", cmd_accuracy))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("thresholds", cmd_thresholds))
    app.add_handler(CommandHandler("elo", cmd_elo))
    return app
