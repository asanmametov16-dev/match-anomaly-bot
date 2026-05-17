"""Основной пайплайн: один цикл опроса API и обработки матчей."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from .config import settings
from .db import Anomaly, OddsSnapshot, SessionLocal
from .detectors import (ALERT_DETECTORS, AnomalyHit, classify_signal,
                        compute_score, detect_cross_market, detect_drift,
                        detect_exotic_spread, detect_model_gap,
                        detect_sharp_move, detect_spread, detect_synchronized,
                        _median)
from .notifier import send_alert
from .odds_client import MatchOdds, fetch_odds
from .probability import consensus_probabilities
from .sstats_client import fetch_xg_batch

log = logging.getLogger(__name__)

DEDUP_WINDOW = timedelta(hours=12)

# Дедуп ОТПРАВКИ: матч → время последнего отправленного алерта. In-memory,
# сбрасывается при рестарте (после рестарта возможен дубль — known
# limitation MVP, см. идею #3 о персистентном дедупе). Это намеренно НЕ
# завязано на сохранённые Anomaly: среди них есть молчаливые weak, которые
# не должны блокировать будущий настоящий сигнал по тому же матчу.
_alerted_at: dict[str, datetime] = {}


def _was_recently_saved(session, match_id: str, detector: str) -> bool:
    """Дедуп ЗАПИСИ: этот детектор по матчу уже сохранён в БД за окно?

    Назначение — не плодить дубль-строки Anomaly каждые 30 мин для
    зреющей часами аномалии. На состав кластера для precision-gate и на
    отправку НЕ влияет (это разные задачи — см. run_once)."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - DEDUP_WINDOW
    count = session.scalar(
        select(func.count(Anomaly.id))
        .where(Anomaly.match_id == match_id)
        .where(Anomaly.detector == detector)
        .where(Anomaly.detected_at >= cutoff)
    )
    return (count or 0) > 0


def _was_match_alerted(match_id: str) -> bool:
    """Дедуп ОТПРАВКИ: по этому матчу уже уходил алерт за окно?"""
    last = _alerted_at.get(match_id)
    if last is None:
        return False
    return datetime.now(timezone.utc).replace(tzinfo=None) - last < DEDUP_WINDOW


def _mark_match_alerted(match_id: str) -> None:
    _alerted_at[match_id] = datetime.now(timezone.utc).replace(tzinfo=None)


def _save_snapshot(session, match: MatchOdds,
                   medians: dict[str, float | None]) -> None:
    session.add(OddsSnapshot(
        match_id=match.match_id,
        sport_key=match.sport_key,
        home_team=match.home_team,
        away_team=match.away_team,
        commence_time=match.commence_time,
        median_home=medians.get("home"),
        median_draw=medians.get("draw"),
        median_away=medians.get("away"),
        bookmakers=[
            {"bookmaker": b.bookmaker, "home": b.home,
             "draw": b.draw, "away": b.away,
             "totals": b.totals, "spreads": b.spreads}
            for b in match.bookmakers
        ],
    ))


def _save_anomaly(session, match: MatchOdds, hit: AnomalyHit) -> None:
    session.add(Anomaly(
        match_id=match.match_id,
        home_team=match.home_team,
        away_team=match.away_team,
        commence_time=match.commence_time,
        detector=hit.detector,
        severity=hit.severity,
        details=hit.description,
        payload=hit.payload,
    ))


async def run_once() -> None:
    """Один цикл: забрать матчи, проверить, сохранить, отправить алерты."""
    log.info("Старт цикла опроса")
    try:
        matches = await fetch_odds()
    except Exception as e:
        log.exception("Не удалось забрать данные с The Odds API: %s", e)
        return

    # Обогащение xG/winProb из sstats.net — батч-вызов до основного цикла.
    # При любой ошибке возвращается пустой dict, model_gap уходит на Elo fallback.
    xg_predictions = await fetch_xg_batch(matches)

    with SessionLocal() as session:
        for match in matches:
            now = datetime.now(match.commence_time.tzinfo)
            hours_until = (match.commence_time - now).total_seconds() / 3600

            # Игнорируем уже сыгранные / стартовавшие
            if hours_until <= 0:
                continue

            # Алертим только матчи в пределах окна: слишком ранние не информативны
            if hours_until > settings.alert_window_hours:
                continue

            # Мало букмекеров → консенсус ненадёжен, снимок не сохраняем
            if len(match.bookmakers) < settings.min_bookmakers_per_match:
                log.debug(
                    "Пропускаем %s vs %s: только %d букмекеров (мин. %d)",
                    match.home_team, match.away_team,
                    len(match.bookmakers), settings.min_bookmakers_per_match,
                )
                continue

            medians = {
                "home": _median(b.home for b in match.bookmakers),
                "draw": _median(b.draw for b in match.bookmakers),
                "away": _median(b.away for b in match.bookmakers),
            }
            # model_gap сравнивает модель с МАРЖА-FREE рынком, а не с
            # сырой медианой коэф. (та несёт overround → ложный gap).
            market_probs = consensus_probabilities(match)

            # Детектируем — порядок важен: drift и synchronized сравнивают с
            # прошлым снимком, поэтому сохранение делаем ПОСЛЕ детекта.
            hits: list[AnomalyHit] = []
            hits += detect_spread(match)
            hits += detect_drift(session, match)
            hits += detect_synchronized(session, match)
            hits += detect_model_gap(session, match, market_probs,
                                     xg_pred=xg_predictions.get(match.match_id))
            hits += detect_sharp_move(match)
            hits += detect_exotic_spread(match)
            hits += detect_cross_market(match)

            # Кластер для precision-gate — ПОЛНЫЙ одновременный набор
            # сработавших детекторов, БЕЗ дедупа. Дедуп ниже касается
            # только записи дублей и повторной отправки; на состав
            # кластера он влиять не должен, иначе аномалия, зреющая за
            # несколько 30-мин циклов, никогда не наберёт MIN_DETECTORS,
            # а записи получат неверный signal_confidence.
            # exotic_spread в ALERT_DETECTORS не входит (шумный) → в gate
            # и алерт не идёт, но сохраняется.
            alert_hits = [h for h in hits if h.detector in ALERT_DETECTORS]

            # Precision-gate по полному кластеру. Метим КАЖДУЮ запись
            # confidence (signal/weak) этим label. Слабые сохраняются, но
            # не эскалируются.
            label, signal_meta = classify_signal(alert_hits)

            # Запись: дедуп дублей — один ряд на (match, detector) за
            # окно. Штамп берём из ПОЛНОГО кластера (label выше), не из
            # подвыборки. Ранее сохранённые ряды отражают кластер на
            # СВОЙ момент и здесь не переписываются (исторически честно).
            for hit in hits:
                if _was_recently_saved(session, match.match_id, hit.detector):
                    continue
                hit.payload = {**(hit.payload or {}), "signal_confidence": label}
                _save_anomaly(session, match, hit)

            _save_snapshot(session, match, medians)

            if settings.signal_gate_enabled:
                do_alert = label == "signal"
            else:
                do_alert = len(alert_hits) >= settings.alert_min_detectors

            # Отправка: не спамим одним и тем же алертом по матчу — дедуп
            # по ФАКТУ отправки, а не по сохранённым аномалиям.
            if (do_alert and alert_hits
                    and not _was_match_alerted(match.match_id)):
                score = compute_score(alert_hits)
                await send_alert(match, alert_hits, score, signal_meta)
                _mark_match_alerted(match.match_id)

        session.commit()

    log.info("Цикл завершён")
