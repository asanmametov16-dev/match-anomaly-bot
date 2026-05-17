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

from .calibration import current_calibration
from .config import settings
from .db import (Anomaly, AnomalyCLV, AnomalyOutcome, OddsSnapshot,
                 ResultNotification, SessionLocal, TeamRating)
from .detectors import DETECTOR_WEIGHTS
from .elo import _normalize as normalize_team
from .prob_calibration import calibration_summary
from .sstats_history import sstats_model_summary

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
        "/signals — только алерты (сильные сигналы): подтв./не подтв.\n"
        "/accuracy — точность детекторов\n"
        "/clv — closing line value по детекторам\n"
        "/calibration — точность вероятностей (Brier)\n"
        "/modelcal — калибровка модели sstats по лигам\n"
        "/weights — веса детекторов (калибровка по CLV)\n"
        "/recent [N] — последние N аномалий\n"
        "/thresholds — текущие пороги\n"
        "/elo <команда> — Elo-рейтинг\n"
        "/help — эта справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


_CONF_MARK = {"signal": "🎯", "weak": "💤"}


def _conf_label(payload: dict | None) -> tuple[str, str]:
    """(маркер, текст) по payload['signal_confidence']. None → нейтрально."""
    c = (payload or {}).get("signal_confidence")
    if c == "signal":
        return "🎯", "signal"
    if c == "weak":
        return "💤", "weak"
    return "·", "—"


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

        # Precision-gate: разбивка записей по signal_confidence
        conf_rows = session.execute(
            select(
                func.json_extract(Anomaly.payload, "$.signal_confidence"),
                func.count(Anomaly.id),
            ).group_by(func.json_extract(Anomaly.payload, "$.signal_confidence"))
        ).all()
        conf_counts = {(k or "—"): c for k, c in conf_rows}

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
        "<b>Precision-gate (записи):</b>",
        f"  🎯 signal: <b>{conf_counts.get('signal', 0)}</b>"
        f"   💤 weak: <b>{conf_counts.get('weak', 0)}</b>"
        f"   · без метки: {conf_counts.get('—', 0)}",
        "",
        "<b>Сигналы (сыгранные матчи):</b>",
        f"  Всего проверено: <b>{sig_total}</b>",
        f"  ✅ Подтвердилось: <b>{sig_confirmed}</b>",
        f"  ❌ Не подтвердилось: <b>{sig_not_confirmed}</b>",
        f"  — Без направления: <b>{sig_no_direction}</b>",
        f"  ❓ Нет результата (лига не в БД): <b>{no_result}</b>",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Только АЛЕРТЫ (сильные сигналы, прошедшие precision-gate): сколько
    было, сколько подтвердилось/не подтвердилось. Общая сводка — /stats."""
    if not _is_authorized(update):
        return
    from .result_checker import _result_key

    with SessionLocal() as session:
        sig_anoms = session.execute(
            select(Anomaly.home_team, Anomaly.away_team,
                   Anomaly.commence_time, Anomaly.detected_at)
            .where(func.json_extract(Anomaly.payload, "$.signal_confidence")
                   == "signal")
        ).all()
        since24 = (datetime.now(timezone.utc).replace(tzinfo=None)
                   - timedelta(hours=24))
        sig_keys: set[str] = set()
        sig_keys_24h: set[str] = set()
        for h, a, ct, det_at in sig_anoms:
            k = _result_key(h, a, ct)
            sig_keys.add(k)
            if det_at and det_at >= since24:
                sig_keys_24h.add(k)
        n_signals = len(sig_keys)

        outcomes = session.execute(select(AnomalyOutcome)).scalars().all()
        by_match: dict[str, list] = {}
        for o in outcomes:
            if o.result_key in sig_keys:
                by_match.setdefault(o.result_key, []).append(o.confirmed)

        confirmed = not_confirmed = no_direction = 0
        for confirmeds in by_match.values():
            directional = [c for c in confirmeds if c is not None]
            if not directional:
                no_direction += 1
            elif sum(1 for c in directional if c == 1) > len(directional) / 2:
                confirmed += 1
            else:
                not_confirmed += 1

        resolved = confirmed + not_confirmed + no_direction
        pending = n_signals - resolved

    if n_signals == 0:
        await update.message.reply_text(
            "📡 <b>Сигналы (алерты)</b>\n\nПока ни одного сигнала не было.",
            parse_mode=ParseMode.HTML)
        return

    decided = confirmed + not_confirmed
    acc = f"{confirmed / decided * 100:.0f}%" if decided else "—"
    lines = [
        "📡 <b>Сигналы (алерты)</b>",
        f"Всего сигналов: <b>{n_signals}</b>  ·  за 24ч: {len(sig_keys_24h)}",
        "",
        f"  ✅ Подтвердилось: <b>{confirmed}</b>",
        f"  ❌ Не подтвердилось: <b>{not_confirmed}</b>",
        f"  ⏳ Ждут результата: <b>{pending}</b>",
        "",
        f"Точность (по сыгранным): <b>{acc}</b>",
    ]
    if no_direction:
        lines.append(f"<i>без направления: {no_direction}</i>")
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


async def cmd_calibration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Калибровка консенсус-вероятностей против реальных исходов."""
    if not _is_authorized(update):
        return

    s = calibration_summary()
    lines = ["🎯 <b>Калибровка вероятностей</b>", ""]

    if s["n"] == 0:
        lines.append("<i>Пока нет оценённых матчей — считается после "
                      "результатов.</i>")
        if s.get("pending"):
            lines.append(f"<i>В очереди без результата: {s['pending']}</i>")
        await update.message.reply_text("\n".join(lines),
                                        parse_mode=ParseMode.HTML)
        return

    brier = s["mean_brier"]
    unif = s["uniform_brier"]
    mark = "🟢" if brier < unif - 0.05 else ("🟡" if brier < unif else "🔴")
    lines += [
        f"матчей: <b>{s['n']}</b>  (в очереди: {s['pending']})",
        f"{mark} Brier: <b>{brier:.4f}</b>  "
        f"<i>(равномерный {unif:.3f}; рынок ≈0.55–0.58)</i>",
        f"log-loss: <b>{s['mean_log_loss']:.4f}</b>",
        "",
        "<b>Кривая надёжности</b> <i>(предсказ. → факт.частота)</i>:",
    ]
    for b in s["bins"]:
        gap = b["emp_freq"] - b["mean_pred"]
        flag = "✓" if abs(gap) < 0.05 else ("↑" if gap > 0 else "↓")
        lines.append(
            f"[{b['lo']:.1f}–{b['hi']:.1f}] n={b['n']:<4d} "
            f"пред={b['mean_pred']:.3f} факт={b['emp_freq']:.3f} {flag}"
        )

    lines += [
        "",
        "<i>Brier &lt; равномерного = рынок информативен. ↑/↓ = бин "
        "недо/переоценён, ✓ = калибровано (±5пп).</i>",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_modelcal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Историческая калибровка модели sstats (Brier по лигам)."""
    if not _is_authorized(update):
        return

    s = sstats_model_summary()
    if not s.get("n"):
        await update.message.reply_text(
            "📊 <b>Калибровка модели sstats</b>\n\n"
            "<i>Нет данных. Запусти бэкфилл: "
            "python -m scripts.backfill_sstats_history</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    unif = s["uniform_brier"]
    b = s["mean_brier"]
    mark = "🟢" if b < unif - 0.05 else ("🟡" if b < unif else "🔴")
    lines = [
        "📊 <b>Калибровка модели sstats</b>",
        "",
        f"матчей: <b>{s['n']}</b>",
        f"{mark} Brier: <b>{b:.4f}</b>  <i>(равномерный {unif:.3f})</i>",
        f"log-loss: <b>{s['mean_log_loss']:.4f}</b>",
        "",
        "<b>По лигам</b> <i>(где модель точнее/хуже)</i>:",
    ]
    for lg in s["leagues"]:
        lb = lg["brier"]
        f = "🟢" if lb < unif - 0.05 else ("🟡" if lb < unif else "🔴")
        lines.append(f"{f} {escape(lg['league'])}: n={lg['n']} Brier={lb:.4f}")
    lines += [
        "",
        "<i>Сравнение с рыночным /calibration → где доверять model_gap.</i>",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Текущие веса детекторов: базовый × CLV-множитель = эффективный."""
    if not _is_authorized(update):
        return

    calib = current_calibration()
    min_n = settings.clv_calibration_min_samples
    enabled = settings.clv_calibration_enabled

    lines = ["⚖️ <b>Веса детекторов</b>", ""]
    if not enabled:
        lines.append("<i>CLV-калибровка выключена — веса базовые.</i>")
        lines.append("")
    lines.append("<i>base × CLV-множитель = эффективный вес</i>")
    lines.append("")

    for det in sorted(DETECTOR_WEIGHTS, key=lambda d: -DETECTOR_WEIGHTS[d]):
        base = DETECTOR_WEIGHTS[det]
        n, mean_clv, mult = calib.get(det, (0, 0.0, 1.0))
        eff = base * mult
        if not enabled or n < min_n:
            tail = f"<i>(n={n}&lt;{min_n}, не калибруется)</i>"
        else:
            tail = f"n={n} ср.CLV={mean_clv:+.2f}пп ×{mult:.2f}"
        lines.append(
            f"<b>{escape(det)}</b>: {base:.1f} → <b>{eff:.2f}</b>  {tail}"
        )

    lines += [
        "",
        f"<i>Множитель = clamp(1 + {settings.clv_calibration_sensitivity:g}×ср.CLV, "
        f"{settings.clv_calibration_min_multiplier:g}, "
        f"{settings.clv_calibration_max_multiplier:g}). "
        f"Пересчёт ежечасно из истории CLV.</i>",
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
        mark, conf = _conf_label(a.payload)
        lines.append(
            f"{mark} <b>{escape(a.home_team)} — {escape(a.away_team)}</b>\n"
            f"  {when} · {a.detector} · severity={a.severity:.2f} · {conf}\n"
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
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("accuracy", cmd_accuracy))
    app.add_handler(CommandHandler("clv", cmd_clv))
    app.add_handler(CommandHandler("calibration", cmd_calibration))
    app.add_handler(CommandHandler("modelcal", cmd_modelcal))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("thresholds", cmd_thresholds))
    app.add_handler(CommandHandler("elo", cmd_elo))
    return app
