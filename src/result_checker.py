"""Джоб проверки результатов матчей с аномалиями и отправки итогов в Telegram."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Anomaly, AnomalyOutcome, MatchResult, ResultNotification, SessionLocal
from .notifier import send_result_message
from .results_client import fetch_finished_matches, normalize_team_name

# Минимальное среднее сходство имён команд для принятия совпадения.
# 0.75 отсекает явно разные клубы, но пропускает варианты написания.
_MATCH_THRESHOLD = 0.75


def _fuzzy_find_result(
    session: Session,
    home: str,
    away: str,
    commence_time: datetime,
) -> MatchResult | None:
    """Нечёткий поиск результата по именам команд и дате.

    Точный поиск по ключу ломается при любом расхождении имён между API.
    Загружаем все MatchResult (таблица маленькая — сотни строк), фильтруем
    по дате ±1 день и выбираем запись с наибольшим средним сходством имён.
    """
    norm_home = normalize_team_name(home)
    norm_away = normalize_team_name(away)
    match_date = commence_time.date()

    all_results = session.execute(select(MatchResult)).scalars().all()

    best_score = 0.0
    best_result: MatchResult | None = None

    for r in all_results:
        # Дата закодирована в result_key как "YYYY-MM-DD|..."
        try:
            r_date = datetime.strptime(r.result_key.split("|")[0], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if abs((r_date - match_date).days) > 1:
            continue

        r_home = normalize_team_name(r.home_team)
        r_away = normalize_team_name(r.away_team)

        score = (
            SequenceMatcher(None, norm_home, r_home).ratio()
            + SequenceMatcher(None, norm_away, r_away).ratio()
        ) / 2

        if score > best_score:
            best_score = score
            best_result = r

    if best_score >= _MATCH_THRESHOLD:
        log.debug("Fuzzy match %.2f: '%s vs %s' → '%s vs %s'",
                  best_score, home, away,
                  best_result.home_team, best_result.away_team)
        return best_result

    log.debug("Fuzzy match failed (best=%.2f) for '%s vs %s'", best_score, home, away)
    return None

log = logging.getLogger(__name__)


def _result_key(home: str, away: str, dt: datetime) -> str:
    return f"{dt.date().isoformat()}|{normalize_team_name(home)}|{normalize_team_name(away)}"


def _backed_outcome(detector: str, payload: dict) -> str | None:
    """Возвращает исход, который детектор считал 'поддержанным рынком'.

    Только для направленных детекторов. spread и exotic_spread направления
    не имеют — возвращаем None.
    """
    if detector == "model_gap":
        # рынок < модели → рынок backing этот исход сильнее Elo
        if payload.get("market", 999.0) < payload.get("fair", 0.0):
            return payload.get("outcome")
    elif detector == "drift":
        # коэф. упал с момента открытия → рынок backing
        if payload.get("current", 999.0) < payload.get("opening", 0.0):
            return payload.get("outcome")
    elif detector == "synchronized":
        # синхронное падение у нескольких контор → рынок backing
        if payload.get("direction") == "↓":
            return payload.get("outcome")
    elif detector == "sharp_move":
        # sharp_prob > soft_prob → профессионалы backing этот исход
        return payload.get("outcome")
    return None


async def check_anomaly_results() -> None:
    """Подтягивает результаты и отправляет итог в Telegram по каждому матчу."""
    finished = await fetch_finished_matches(days_back=3)
    if not finished:
        return

    with SessionLocal() as session:
        # Сохраняем свежие результаты в БД
        for m in finished:
            key = _result_key(m.home_team, m.away_team, m.utc_date)
            if session.get(MatchResult, key) is None:
                session.add(MatchResult(
                    result_key=key,
                    home_team=m.home_team,
                    away_team=m.away_team,
                    home_score=m.home_score,
                    away_score=m.away_score,
                    competition=m.competition,
                ))
        session.commit()

        # Ищем аномалии завершившихся матчей (>2ч с начала)
        cutoff = datetime.utcnow() - timedelta(hours=2)
        anomalies = session.execute(
            select(Anomaly).where(Anomaly.commence_time <= cutoff)
        ).scalars().all()

        if not anomalies:
            return

        # Группируем по матчу
        by_match: dict[str, list[Anomaly]] = {}
        for a in anomalies:
            key = _result_key(a.home_team, a.away_team, a.commence_time)
            by_match.setdefault(key, []).append(a)

        for result_key, match_anomalies in by_match.items():
            if session.get(ResultNotification, result_key) is not None:
                continue  # уже отправляли

            first = match_anomalies[0]
            result = _fuzzy_find_result(
                session, first.home_team, first.away_team, first.commence_time
            )
            if result is None:
                # После 96 часов с начала матча прекращаем попытки — лига скорее всего
                # вне покрытия football-data.org (MLS, Süper Lig, Eredivisie и т.д.)
                age_hours = (datetime.utcnow() - first.commence_time.replace(tzinfo=None)
                             ).total_seconds() / 3600
                if age_hours > 96:
                    log.warning(
                        "Результат '%s vs %s' (%s) не найден за 96ч — "
                        "лига, вероятно, вне football-data.org. Помечаем как проверено.",
                        first.home_team, first.away_team,
                        first.commence_time.date(),
                    )
                    session.add(ResultNotification(result_key=result_key, result_found=False))
                else:
                    log.debug("Результат '%s vs %s' пока не найден (%.0fч после матча)",
                              first.home_team, first.away_team, age_hours)
                continue

            # Фактический исход
            if result.home_score > result.away_score:
                actual = "home"
            elif result.away_score > result.home_score:
                actual = "away"
            else:
                actual = "draw"

            # Вердикт по каждому детектору
            confirmed_flags: list[bool] = []
            detector_verdicts: list[tuple[str, bool | None]] = []
            seen_detectors: set[str] = set()

            for a in sorted(match_anomalies, key=lambda x: -x.severity):
                if a.detector in seen_detectors:
                    continue
                seen_detectors.add(a.detector)
                backed = _backed_outcome(a.detector, a.payload or {})
                if backed is not None:
                    c = backed == actual
                    confirmed_flags.append(c)
                    detector_verdicts.append((a.detector, c))
                else:
                    detector_verdicts.append((a.detector, None))

            overall: bool | None = None
            if confirmed_flags:
                # Большинство голосов: >50% детекторов подтвердили сигнал
                yes = sum(1 for c in confirmed_flags if c)
                overall = yes > len(confirmed_flags) / 2

            await send_result_message(
                home_team=first.home_team,
                away_team=first.away_team,
                home_score=result.home_score,
                away_score=result.away_score,
                commence_time=first.commence_time,
                competition=result.competition or "",
                detector_verdicts=detector_verdicts,
                overall=overall,
            )

            # Сохраняем вердикты для статистики
            for detector, confirmed in detector_verdicts:
                session.add(AnomalyOutcome(
                    result_key=result_key,
                    detector=detector,
                    confirmed=(1 if confirmed is True else (0 if confirmed is False else None)),
                ))
            session.add(ResultNotification(result_key=result_key))

        session.commit()

    log.info("Проверка результатов завершена")
