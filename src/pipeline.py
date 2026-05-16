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
from .sstats_client import fetch_xg_batch

log = logging.getLogger(__name__)

DEDUP_WINDOW = timedelta(hours=12)


def _was_recently_alerted(session, match_id: str, detector: str) -> bool:
    """Проверяет БД: было ли срабатывание этого детектора по матчу за последние 12ч."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - DEDUP_WINDOW
    count = session.scalar(
        select(func.count(Anomaly.id))
        .where(Anomaly.match_id == match_id)
        .where(Anomaly.detector == detector)
        .where(Anomaly.detected_at >= cutoff)
    )
    return (count or 0) > 0


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

            # Детектируем — порядок важен: drift и synchronized сравнивают с
            # прошлым снимком, поэтому сохранение делаем ПОСЛЕ детекта.
            hits: list[AnomalyHit] = []
            hits += detect_spread(match)
            hits += detect_drift(session, match)
            hits += detect_synchronized(session, match)
            hits += detect_model_gap(session, match, medians,
                                     xg_pred=xg_predictions.get(match.match_id))
            hits += detect_sharp_move(match)
            hits += detect_exotic_spread(match)
            hits += detect_cross_market(match)

            # Дедупликация: не отправляем повторно срабатывание того же
            # детектора по тому же матчу в течение DEDUP_WINDOW
            fresh_hits = [
                h for h in hits
                if not _was_recently_alerted(session, match.match_id, h.detector)
            ]

            # exotic_spread / cross_market сохраняются в БД, но в алерт не идут
            alert_hits = [h for h in fresh_hits if h.detector in ALERT_DETECTORS]

            # Precision-gate: классифицируем кластер и метим КАЖДУЮ запись
            # confidence (signal/weak). Слабые сохраняются, но не эскалируются.
            label, signal_meta = classify_signal(alert_hits)
            for hit in fresh_hits:
                hit.payload = {**(hit.payload or {}), "signal_confidence": label}
                _save_anomaly(session, match, hit)

            _save_snapshot(session, match, medians)

            if settings.signal_gate_enabled:
                do_alert = label == "signal"
            else:
                do_alert = len(alert_hits) >= settings.alert_min_detectors

            if do_alert and alert_hits:
                score = compute_score(alert_hits)
                await send_alert(match, alert_hits, score, signal_meta)

        session.commit()

    log.info("Цикл завершён")
