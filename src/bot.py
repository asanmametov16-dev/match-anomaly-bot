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
from datetime import datetime, timedelta, timezone
from html import escape

from sqlalchemy import desc, func, select
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          filters)

from .config import settings
from .db import (Anomaly, AnomalyCLV, AnomalyOutcome, OddsSnapshot,
                 ResultNotification, SessionLocal, TeamRating)
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
        "/clv — closing line value по детекторам\n"
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
            .where(Anomaly.detected_at >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24))
        ) or 0
        snapshots = session.scalar(select(func.count(OddsSnapshot.id))) or 0
        teams = session.scalar(select(func.count(TeamRating.team))) or 0

        by_detector = session.execute(
            select(Anomaly.detector, func.count(Anomaly.id))
            .group_by(Anomaly.detector)
        ).all()

        # Статистика сигналов по результатам
        outcomes = session.execute(select(AnomalyOutcome)).scalars().all()
        no_result = session.scalar(
            select(func.count(ResultNotification.result_key))
            .where(ResultNotification.result_found == False)  # noqa: E712
        ) or 0

        # Группируем AnomalyOutcome по матчу и считаем большинством голосов
        by_match: dict[str, list] = {}
        for o in outcomes:
            by_match.setdefault(o.result_key, []).append(o.confirmed)

        sig_confirmed = sig_not_confirmed = sig_no_direction = 0
        for confirmeds in by_match.values():
            directional = [c for c in confirmeds if c is not None]
            if not directional:
                sig_no_direction += 1
            elif sum(1 for c in directional if c == 1) > len(directional) / 2:
                sig_confirmed += 1
            else:
                sig_not_confirmed += 1

        sig_total = sig_confirmed + sig_not_confirmed + sig_no_direction + no_result

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

    lines += [
        "",
        "<b>Сигналы (сыгранные матчи):</b>",
        f"  Всего проверено: <b>{sig_total}</b>",
        f"  ✅ Подтвердилось: <b>{sig_confirmed}</b>",
        f"  ❌ Не подтвердилось: <b>{sig_not_confirmed}</b>",
        f"  — Без направления: <b>{sig_no_direction}</b>",
        f"  ❓ Нет результата (лига не в БД): <b>{no_result}</b>",
    ]

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


async def cmd_clv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Closing line value по детекторам — единственная честная метрика сигнала."""
    if not _is_authorized(update):
        return

    with SessionLocal() as session:
        rows = session.execute(
            select(AnomalyCLV).where(AnomalyCLV.clv_pp.isnot(None))
        ).scalars().all()
        non_dir = session.scalar(
            select(func.count(AnomalyCLV.anomaly_id))
            .where(AnomalyCLV.side.is_(None))
        ) or 0
        no_data = session.scalar(
            select(func.count(AnomalyCLV.anomaly_id))
            .where(AnomalyCLV.side.isnot(None))
            .where(AnomalyCLV.clv_pp.is_(None))
        ) or 0

    by_detector: dict[str, list[float]] = {}
    for r in rows:
        by_detector.setdefault(r.detector, []).append(r.clv_pp)

    lines = ["📈 <b>CLV (closing line value)</b>", ""]
    if not by_detector:
        lines.append("<i>Пока нет данных — CLV считается после старта матча.</i>")
    else:
        lines.append("<i>+CLV: рынок двинулся дальше в сторону прогноза.</i>")
        lines.append("")
        for det, vals in sorted(by_detector.items(), key=lambda kv: -len(kv[1])):
            n = len(vals)
            mean = sum(vals) / n
            pos_pct = sum(1 for v in vals if v > 0) / n * 100
            if n >= 2:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                sigma_str = f"σ={var ** 0.5:.1f}"
            else:
                sigma_str = "σ=—"
            if n < 10:
                marker, note = "⚪", "  <i>(n&lt;10, мало)</i>"
            elif mean > 0.5:
                marker, note = "🟢", ""
            elif mean < -0.5:
                marker, note = "🔴", ""
            else:
                marker, note = "🟡", ""
            lines.append(
                f"{marker} <b>{escape(det)}</b>: "
                f"n={n}  mean={mean:+.2f}пп  pos%={pos_pct:.0f}%  {sigma_str}{note}"
            )

    lines += [
        "",
        f"<i>Без CLV: {non_dir} не-направленных + {no_data} без снимков</i>",
        "<i>🟢 mean&gt;+0.5пп при n≥10  🟡 около нуля  🔴 mean&lt;−0.5пп</i>",
        "<i>Положительный CLV — необходимое, но не достаточное условие "
        "прибыльности (нужно ещё перекрыть маржу букмекера и vig).</i>",
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
    app.add_handler(CommandHandler("clv", cmd_clv))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("thresholds", cmd_thresholds))
    app.add_handler(CommandHandler("elo", cmd_elo))
    return app
