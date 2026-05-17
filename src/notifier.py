"""Отправка уведомлений в Telegram.

Использует тот же Bot, что и команды бота — это важно, чтобы не плодить
параллельные сессии. Bot устанавливается из main.py через set_bot().
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from html import escape

from telegram import Bot
from telegram.constants import ParseMode

from .config import settings
from .detectors import AnomalyHit
from .odds_client import MatchOdds

log = logging.getLogger(__name__)

_bot: Bot | None = None


def set_bot(bot: Bot) -> None:
    global _bot
    _bot = bot


_OUTCOME_RU = {"home": "хозяева", "away": "гости", "draw": "ничья"}
# Дательный падеж — для «по X» в описаниях детекторов
_OUTCOME_RU_DAT = {"home": "хозяевам", "away": "гостям", "draw": "ничьей"}
# Винительный падеж — для «оценивает X фаворитом»
_OUTCOME_RU_ACC = {"home": "хозяев", "away": "гостей", "draw": "ничью"}

_DETECTOR_META: dict[str, tuple[str, str]] = {
    "spread": (
        "Разброс между букмекерами",
        "Часть контор ещё не обновила линию — вероятно, стало известно что-то новое "
        "(состав, травма, погода). Разрыв закроется, когда все конторы среагируют.",
    ),
    "drift": (
        "Движение линии с момента открытия",
        "Коэффициент резко сдвинулся от первоначального значения — признак крупных ставок "
        "или важных новостей после открытия рынка.",
    ),
    "synchronized": (
        "Синхронное движение у нескольких контор",
        "3+ букмекера одновременно сдвинули линию в одну сторону — сильный сигнал "
        "\"умных денег\": кто-то ставит сразу в несколько мест.",
    ),
    "model_gap": (
        "Рынок расходится с моделью",
        "",  # генерируется динамически (_model_gap_explanation, по source)
    ),
    "exotic_spread": (
        "Разброс на тоталах / форах",
        "На вспомогательных рынках (тоталы, форы) конторы не синхронизированы — "
        "эти рынки менее ликвидны и медленнее обновляются.",
    ),
    "sharp_move": (
        "Sharp-движение (Pinnacle / Betfair vs остальные)",
        "Острые букмекеры дают заметно более низкий коэф., чем мягкие конторы — "
        "признак того, что профессиональные игроки уже поставили на этот исход.",
    ),
    "cross_market": (
        "Межрыночное расхождение (1X2 vs фора 0.0)",
        "Фора 0.0 = Draw-No-Bet, её вероятность обязана совпадать с 1X2 по "
        "тождеству. Расхождение — признак устаревшей линии или ошибки в одном "
        "из рынков. Не указывает сторону, лишь усиливает кластер.",
    ),
}


# Прежние _implied_probabilities / _backed_outcome_from_hit /
# _anomaly_adjusted_probabilities удалены: «скорректированная вероятность»
# суммировала несопоставимые severity и порождала EV-вердикт (ставочная
# рамка). Направление теперь — единственный источник: precision-gate
# (meta) + extract_bet_side. Блок вероятностей показывает честный
# маржа-free консенсус рынка (см. _format_probabilities).


def _format_signal(hits: list[AnomalyHit], match: MatchOdds,
                   meta: dict | None = None,
                   score: float | None = None) -> str:
    """Формирует строку с направлением сигнала.

    Источник истины — precision-gate (`classify_signal`/meta): именно он
    решил эскалировать алерт. Если meta задан, берём сторону из него и НЕ
    выводим независимый вердикт «противоречивый» (гейт уже гарантировал
    согласие ≥ порога — иначе алерта бы не было). Без meta — старая
    severity-эвристика как fallback. Сторона детектора определяется тем же
    extract_bet_side, что и в гейте, — чтобы не расходиться.
    """
    from .clv import extract_bet_side

    outcome_names = {
        "home": f"Победа хозяев ({escape(match.home_team)})",
        "away": f"Победа гостей ({escape(match.away_team)})",
        "draw": "Ничья",
    }

    if meta and meta.get("side") in outcome_names:
        # Источник истины — precision-gate. Сила выводится ИЗ ГЕЙТА
        # (CLV-взвешенный score + число направленных детекторов), теми
        # же порогами, что «уверенность» в шапке, — а не из суммы
        # несопоставимых severity (drift в пп ≫ sharp_move в долях).
        best = meta["side"]
        nd = int(meta.get("side_detectors", 0) or 0)
        agree = float(meta.get("agreement", 0.0) or 0.0)
        if (score is not None and score >= 5) or nd >= 3:
            strength, icon = "сильный", "🔥"
        elif (score is not None and score >= 3) or nd >= 2:
            strength, icon = "средний", "⚡"
        else:
            strength, icon = "слабый", "💧"
        return (
            f"📌 <b>Сигнал:</b> {icon} {outcome_names[best]} — <b>{strength}</b>\n"
            f"   <i>(направленных за сторону: {nd}, согласие {agree*100:.0f}%)</i>"
        )

    # Fallback без гейта (signal_gate_enabled=false): severity-эвристика.
    sev: dict[str, float] = {"home": 0.0, "draw": 0.0, "away": 0.0}
    cnt: dict[str, int] = {"home": 0, "draw": 0, "away": 0}
    for hit in hits:
        backed = extract_bet_side(hit.detector, hit.payload or {})
        if backed and backed in sev:
            sev[backed] += hit.severity
            cnt[backed] += 1

    total_sev = sum(sev.values())
    if total_sev == 0:
        return ""  # только ненаправленные детекторы
    best = max(sev, key=lambda k: sev[k])
    if sev[best] / total_sev < 0.55:
        return ("📌 <b>Сигнал:</b> <i>противоречивый — детекторы "
                "указывают в разные стороны</i>")

    n = cnt[best]
    if sev[best] >= 0.6 or n >= 3:
        strength, icon = "сильный", "🔥"
    elif sev[best] >= 0.3 or n >= 2:
        strength, icon = "средний", "⚡"
    else:
        strength, icon = "слабый", "💧"
    return (
        f"📌 <b>Сигнал:</b> {icon} {outcome_names[best]} — <b>{strength}</b>\n"
        f"   <i>({n} детектор(а), суммарная сила {sev[best]:.2f})</i>"
    )


def _best_odds(match: MatchOdds) -> dict[str, float]:
    """Лучший (максимальный) доступный коэффициент по каждому исходу."""
    result: dict[str, float] = {}
    for outcome in ("home", "draw", "away"):
        prices = [getattr(b, outcome) for b in match.bookmakers]
        clean = [p for p in prices if p is not None and p > 1.0]
        if clean:
            result[outcome] = max(clean)
    return result


def _format_probabilities(match: MatchOdds) -> str:
    """Честный рыночный расклад: МАРЖА-FREE консенсус вероятностей + лучший
    доступный коэффициент по каждому исходу.

    Никаких «скорректированных» вероятностей, EV и stake-рамки — это
    аналитический сигнал, а не ставочная рекомендация (см. CLAUDE.md /
    _SIGNAL_DISCLAIMER). Цифра справочная: что рынок думает об исходе
    после снятия маржи (метод из probability.py).
    """
    from .probability import consensus_probabilities

    cp = consensus_probabilities(match)
    if not cp:
        return ""
    s = sum(v for v in cp.values() if v and v > 0.0)
    if s <= 0.0:
        return ""
    cp = {k: v / s for k, v in cp.items() if v and v > 0.0}

    best = _best_odds(match)
    outcome_names = {
        "home": f"Победа хозяев ({escape(match.home_team)})",
        "away": f"Победа гостей ({escape(match.away_team)})",
        "draw": "Ничья",
    }
    medals = ["🥇", "🥈", "🥉"]

    lines = []
    for i, (outcome, prob) in enumerate(
        sorted(cp.items(), key=lambda kv: -kv[1])
    ):
        medal = medals[i] if i < len(medals) else "  "
        bo = best.get(outcome)
        odds_str = f"  (лучший коэф: {bo:.2f})" if bo else ""
        lines.append(
            f"{medal} {outcome_names.get(outcome, outcome)}: "
            f"<b>{prob * 100:.0f}%</b>{odds_str}"
        )
    return "\n".join(lines)


def _model_gap_explanation(group: list[AnomalyHit]) -> str:
    """Текст ветвится по payload['source']: модель у model_gap — это
    sstats-xG / консенсус sharp-контор / Elo (см. detect_model_gap).
    Раньше всегда писалось «Elo-модель» — после перевода на маржа-free
    это было фактически неверно для xg/sharp."""
    source = (group[0].payload.get("source") if group else "") or ""
    model_name = {
        "sstats_xg": "xG-моделью sstats",
        "sharp_consensus": "консенсусом sharp-контор",
        "elo": "Elo-моделью",
    }.get(source, "моделью")
    if source == "elo":
        caveat = (" Elo не учитывает свежие данные (травма, смена тренера, "
                  "серия результатов) и шумит первые 1–2 недели после старта.")
    elif source == "sharp_consensus":
        caveat = (" Референс — медиана sharp-контор: расхождение значит, что "
                  "общий рынок ещё не подтянулся к острым деньгам.")
    elif source == "sstats_xg":
        caveat = (" Референс — xG/winProb sstats: рынок оценивает матч иначе, "
                  "чем модель по ожидаемым голам.")
    else:
        caveat = ""

    market_lower: list[str] = []  # market<fair → рынок ценит исход выше модели
    for hit in group:
        outcome = hit.payload.get("outcome", "")
        if hit.payload.get("market", 0.0) < hit.payload.get("fair", 0.0):
            market_lower.append(_OUTCOME_RU_ACC.get(outcome, outcome))
    if market_lower:
        favored = " и ".join(market_lower)
        return (f"Рынок оценивает {favored} фаворитом заметно сильнее, чем "
                f"{model_name}.{caveat}")
    return f"Рынок и оценка {model_name} расходятся в этом матче.{caveat}"


_SIGNAL_DISCLAIMER = (
    "ℹ️ <i>Аналитический сигнал рыночной аномалии — не ставочная и не "
    "инвестиционная рекомендация. Решение и риск остаются за вами.</i>"
)


_SIDE_RU = {"home": "П1 (хозяева)", "draw": "Х (ничья)", "away": "П2 (гости)"}


def _format_gate_line(meta: dict | None) -> str:
    """Компактная строка-обоснование precision-gate (почему это сигнал)."""
    if not meta:
        return ""
    side = _SIDE_RU.get(meta.get("side") or "", "—")
    sd = meta.get("side_detectors", 0)
    nd = meta.get("n_directional", 0)
    parts = [
        f"🧭 Сторона: <b>{side}</b>",
        f"за неё {sd}/{nd} направл. (согласие "
        f"{meta.get('agreement', 0) * 100:.0f}%)",
        f"детекторов всего {meta.get('n_detectors', 0)}",
    ]
    dropped = meta.get("dropped_untrusted_model_gap", 0)
    if dropped:
        parts.append(f"⚠️ отсеян model_gap×{dropped} (ненадёжная лига)")
    return " · ".join(parts)


def _format_message(match: MatchOdds, hits: list[AnomalyHit], score: float,
                    meta: dict | None = None) -> str:
    if score >= 5:
        level_icon, level_text = "🎯", "высокая уверенность"
    elif score >= 3:
        level_icon, level_text = "📈", "средняя уверенность"
    else:
        level_icon, level_text = "💡", "низкая уверенность"

    lines = [
        f"{level_icon} <b>Рыночный сигнал</b>  [{level_text}]",
        f"<b>{escape(match.home_team)}</b> — <b>{escape(match.away_team)}</b>",
        f"🕐 {(match.commence_time.replace(tzinfo=timezone.utc) + timedelta(hours=3)).strftime('%d.%m.%Y %H:%M')} МСК",
        f"🏆 {escape(match.sport_key)}",
        "",
        f"Сила сигнала: <b>{score:.1f}</b>  |  Детекторов: {len(hits)}",
    ]
    gate_line = _format_gate_line(meta)
    if gate_line:
        lines.append(gate_line)
    lines.append("")

    by_detector: dict[str, list[AnomalyHit]] = defaultdict(list)
    for hit in sorted(hits, key=lambda h: -h.severity):
        by_detector[hit.detector].append(hit)

    for detector, group in by_detector.items():
        title, explanation = _DETECTOR_META.get(detector, (detector, ""))
        lines.append(f"🔍 <b>{escape(title)}</b>")
        for hit in group:
            desc = hit.description
            for en, ru in _OUTCOME_RU_DAT.items():
                desc = desc.replace(f" {en}:", f" {ru}:")
            lines.append(f"  • {escape(desc)}")
        if detector == "model_gap":
            explanation = _model_gap_explanation(group)
        if explanation:
            lines.append(f"  💡 <i>{escape(explanation)}</i>")
        lines.append("")

    signal = _format_signal(hits, match, meta, score)
    if signal:
        lines += ["", signal]

    prob_block = _format_probabilities(match)
    if prob_block:
        lines += ["", "<b>Рыночная оценка (маржа-free консенсус):</b>",
                  prob_block]

    lines += ["", _SIGNAL_DISCLAIMER]
    return "\n".join(lines)


async def send_alert(match: MatchOdds, hits: list[AnomalyHit], score: float = 0.0,
                     meta: dict | None = None) -> None:
    if not hits or _bot is None:
        return
    text = _format_message(match, hits, score, meta)
    try:
        await _bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
        log.info("Алерт отправлен: %s vs %s, детекторов: %d",
                 match.home_team, match.away_team, len(hits))
    except Exception as e:
        log.error("Не удалось отправить алерт: %s", e)


async def send_result_message(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    commence_time: datetime,
    competition: str,
    detector_verdicts: list[tuple[str, bool | None]],
    overall: bool | None,
) -> None:
    if _bot is None:
        return

    date_msk = (commence_time.replace(tzinfo=timezone.utc) + timedelta(hours=3)).strftime("%d.%m.%Y")

    if overall is True:
        header = "✅ <b>Аномалия подтвердилась</b>"
        note = "Рынок сигнализировал верно — линия двигалась в сторону победителя."
    elif overall is False:
        header = "❌ <b>Аномалия не подтвердилась</b>"
        note = "Сигнал оказался ложным — результат не совпал с направлением рынка."
    else:
        header = "📊 <b>Результат матча</b>"
        note = "Направленных детекторов не было — вердикт не определяется."

    lines = [
        header,
        f"<b>{escape(home_team)}</b> — <b>{escape(away_team)}</b>",
        f"Счёт: <b>{home_score} : {away_score}</b>",
        f"🕐 {date_msk}   🏆 {escape(competition)}",
        "",
        "Детекторы:",
    ]

    for detector, confirmed in detector_verdicts:
        icon = "✅" if confirmed is True else ("❌" if confirmed is False else "—")
        title = _DETECTOR_META.get(detector, (detector, ""))[0]
        lines.append(f"  {icon} {escape(title)}")

    lines += ["", f"<i>{escape(note)}</i>"]

    try:
        await _bot.send_message(
            chat_id=settings.telegram_chat_id,
            text="\n".join(lines),
            parse_mode=ParseMode.HTML,
        )
        log.info("Результат отправлен: %s vs %s %d:%d overall=%s",
                 home_team, away_team, home_score, away_score, overall)
    except Exception as e:
        log.error("Не удалось отправить результат: %s", e)


async def send_startup_message() -> None:
    if _bot is None:
        return
    text = (
        f"🤖 Anomaly bot запущен в {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
        f"Опрос каждые {settings.poll_interval_minutes} мин\n"
        f"Спорт: {settings.odds_api_sport}, регионы: {settings.odds_api_regions}\n"
        f"Команды: /help"
    )
    try:
        await _bot.send_message(chat_id=settings.telegram_chat_id, text=text)
    except Exception as e:
        log.error("Не удалось отправить стартовое сообщение: %s", e)
