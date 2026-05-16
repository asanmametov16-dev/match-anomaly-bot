"""Отправка уведомлений в Telegram.

Использует тот же Bot, что и команды бота — это важно, чтобы не плодить
параллельные сессии. Bot устанавливается из main.py через set_bot().
"""
from __future__ import annotations

import logging
import statistics
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
        "Рынок расходится с Elo-моделью",
        "",  # генерируется динамически
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
}


def _implied_probabilities(match: MatchOdds) -> list[tuple[str, float, float]]:
    """Нормализованные вероятности исходов по медианным коэффициентам.

    Возвращает список (outcome, probability, median_odds), отсортированный
    по убыванию вероятности.
    """
    medians: dict[str, float] = {}
    for outcome in ("home", "draw", "away"):
        prices = [getattr(b, outcome) for b in match.bookmakers]
        clean = [p for p in prices if p is not None and p > 1.0]
        if clean:
            medians[outcome] = statistics.median(clean)

    if not medians:
        return []

    raw = {k: 1.0 / v for k, v in medians.items()}
    total = sum(raw.values())
    return sorted(
        [(k, raw[k] / total, medians[k]) for k in raw],
        key=lambda x: -x[1],
    )


def _backed_outcome_from_hit(hit: AnomalyHit) -> str | None:
    """Исход, который детектор считает 'поддержанным рынком'."""
    payload = hit.payload or {}
    if hit.detector == "model_gap":
        if payload.get("market", 999.0) < payload.get("fair", 0.0):
            return payload.get("outcome")
    elif hit.detector == "drift":
        if payload.get("current", 999.0) < payload.get("opening", 0.0):
            return payload.get("outcome")
    elif hit.detector == "synchronized":
        if payload.get("direction") == "↓":
            return payload.get("outcome")
    elif hit.detector == "sharp_move":
        # sharp_prob > soft_prob → sharps backing этот исход
        if payload.get("sharp_prob", 0.0) > payload.get("soft_prob", 0.0):
            return payload.get("outcome")
    return None


def _anomaly_adjusted_probabilities(
    match: MatchOdds, hits: list[AnomalyHit]
) -> list[tuple[str, float, float, float]]:
    """Вероятности, скорректированные на сигналы аномалий.

    Алгоритм:
    1. Базовые вероятности из медианных коэффициентов (с нормализацией маржи).
    2. Каждый направленный детектор увеличивает вероятность «поддержанного»
       исхода пропорционально severity (насколько сильна аномалия).
    3. Перенормировка.

    Возвращает (outcome, adjusted_prob, base_prob, median_odds).
    """
    base = _implied_probabilities(match)
    if not base:
        return []

    base_probs = {o: p for o, p, _ in base}
    odds_map = {o: od for o, _, od in base}

    boost: dict[str, float] = {k: 0.0 for k in base_probs}
    for hit in hits:
        backed = _backed_outcome_from_hit(hit)
        if backed and backed in boost:
            boost[backed] += hit.severity

    adjusted = {k: base_probs[k] * (1.0 + boost[k]) for k in base_probs}
    total = sum(adjusted.values())
    normalized = {k: v / total for k, v in adjusted.items()}

    return sorted(
        [(k, normalized[k], base_probs[k], odds_map[k]) for k in normalized],
        key=lambda x: -x[1],
    )


def _format_signal(hits: list[AnomalyHit], match: MatchOdds,
                   meta: dict | None = None) -> str:
    """Формирует строку с направлением сигнала.

    Источник истины — precision-gate (`classify_signal`/meta): именно он
    решил эскалировать алерт. Если meta задан, берём сторону из него и НЕ
    выводим независимый вердикт «противоречивый» (гейт уже гарантировал
    согласие ≥ порога — иначе алерта бы не было). Без meta — старая
    severity-эвристика как fallback. Сторона детектора определяется тем же
    extract_bet_side, что и в гейте, — чтобы не расходиться.
    """
    from .clv import extract_bet_side

    score: dict[str, float] = {"home": 0.0, "draw": 0.0, "away": 0.0}
    detector_count: dict[str, int] = {"home": 0, "draw": 0, "away": 0}

    for hit in hits:
        backed = extract_bet_side(hit.detector, hit.payload or {})
        if backed and backed in score:
            score[backed] += hit.severity
            detector_count[backed] += 1

    total_score = sum(score.values())

    if meta and meta.get("side") in score:
        # Авторитетная сторона от гейта — без противоречий с gate-строкой.
        best = meta["side"]
        best_score = score[best]
    else:
        if total_score == 0:
            return ""  # только ненаправленные детекторы
        best = max(score, key=lambda k: score[k])
        best_score = score[best]
        # Вердикт «противоречивый» — только в fallback без гейта.
        if best_score / total_score < 0.55:
            return ("📌 <b>Сигнал:</b> <i>противоречивый — детекторы "
                    "указывают в разные стороны</i>")

    outcome_names = {
        "home": f"Победа хозяев ({escape(match.home_team)})",
        "away": f"Победа гостей ({escape(match.away_team)})",
        "draw": "Ничья",
    }

    # Сила сигнала
    n = detector_count[best]
    if best_score >= 0.6 or n >= 3:
        strength, icon = "сильный", "🔥"
    elif best_score >= 0.3 or n >= 2:
        strength, icon = "средний", "⚡"
    else:
        strength, icon = "слабый", "💧"

    return (
        f"📌 <b>Сигнал:</b> {icon} {outcome_names[best]} — <b>{strength}</b>\n"
        f"   <i>({n} детектор(а), суммарная сила {best_score:.2f})</i>"
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


def _format_probabilities(match: MatchOdds, hits: list[AnomalyHit]) -> str:
    results = _anomaly_adjusted_probabilities(match, hits)
    if not results:
        return ""

    best = _best_odds(match)
    outcome_names = {
        "home": f"Победа хозяев ({escape(match.home_team)})",
        "away": f"Победа гостей ({escape(match.away_team)})",
        "draw": "Ничья",
    }
    medals = ["🥇", "🥈", "🥉"]

    lines = []
    for i, (outcome, adj_prob, base_prob, median_odds) in enumerate(results):
        medal = medals[i] if i < len(medals) else "  "

        diff = adj_prob - base_prob
        trend = ""
        if diff > 0.02:
            trend = f" ↑{diff*100:.0f}пп"
        elif diff < -0.02:
            trend = f" ↓{abs(diff)*100:.0f}пп"

        best_odd = best.get(outcome, median_odds)
        ev = adj_prob * best_odd - 1
        if ev > 0.03:
            ev_str = f"  ✅ EV <b>+{ev*100:.1f}%</b>"
        elif ev > 0:
            ev_str = f"  ⚠️ EV +{ev*100:.1f}%"
        else:
            ev_str = f"  ❌ EV {ev*100:.1f}%"

        lines.append(
            f"{medal} {outcome_names[outcome]}: "
            f"<b>{adj_prob * 100:.0f}%</b>{escape(trend)}"
            f"  (лучший коэф: {best_odd:.2f})"
            f"{ev_str}"
        )
    return "\n".join(lines)


def _model_gap_explanation(group: list[AnomalyHit]) -> str:
    market_lower: list[str] = []  # рынок ниже модели → рынок считает фаворитом
    for hit in group:
        outcome = hit.payload.get("outcome", "")
        market = hit.payload.get("market", 0.0)
        fair = hit.payload.get("fair", 0.0)
        if market < fair:
            market_lower.append(_OUTCOME_RU_ACC.get(outcome, outcome))
    if market_lower:
        favored = " и ".join(market_lower)
        return (
            f"Рынок оценивает {favored} фаворитом значительно сильнее, чем Elo-модель. "
            "Возможные причины: Elo не учитывает свежие данные (травма, смена тренера, "
            "серия результатов). Модель шумит первые 1–2 недели после старта."
        )
    return (
        "Рынок и Elo-модель расходятся в оценке матча. "
        "Модель шумит первые 1–2 недели, пока рейтинги не наберут статистику."
    )


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
    parts = [
        f"🧭 Сторона: <b>{side}</b>",
        f"согласие {meta.get('agreement', 0) * 100:.0f}%",
        f"детекторов {meta.get('n_detectors', 0)}",
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

    signal = _format_signal(hits, match, meta)
    if signal:
        lines += ["", signal]

    prob_block = _format_probabilities(match, hits)
    if prob_block:
        lines += ["", "<b>Вероятные исходы событий:</b>", prob_block]

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
