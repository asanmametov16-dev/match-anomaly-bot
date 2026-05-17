"""Джоб проверки результатов матчей с аномалиями и отправки итогов в Telegram."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from .clv import extract_bet_side
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


# Направление = ЕДИНЫЙ источник правды: clv.extract_bet_side (тот же,
# что у precision-gate и CLV). Раньше тут была локальная копия
# _backed_outcome, которая для model_gap указывала ПРОТИВОПОЛОЖНУЮ
# сторону (market<fair вместо value-side market>fair), а для drift
# читала несуществующие ключи (current/opening вместо *_prob/drift_pp)
# → /accuracy мерил инвертированную гипотезу и терял drift целиком.


def clear_false_sentinels() -> int:
    """Удалить ResultNotification с result_found=False (ложные «сдались за
    96ч»). Нужно для catch-up: широкое окно/второй источник могут теперь
    закрыть матчи, по которым ранее не нашли результат. Если матч всё ещё
    не резолвится — check_anomaly_results заново поставит sentinel через 96ч.
    Возвращает число удалённых. Идемпотентно.
    """
    with SessionLocal() as session:
        stale = session.execute(
            select(ResultNotification).where(
                ResultNotification.result_found == False)  # noqa: E712
        ).scalars().all()
        for rn in stale:
            session.delete(rn)
        session.commit()
        return len(stale)


async def check_anomaly_results(days_back: int = 3,
                                sstats_max_pages: int = 8) -> None:
    """Подтягивает результаты и отправляет итог в Telegram по каждому матчу.

    days_back/sstats_max_pages по умолчанию узкие (ежечасный джоб); catch-up
    вызывает с широким окном для ретро-резолва старого бэклога.
    """
    # Два источника: football-data.org + sstats (покрывает лиги вне FD).
    # Локальный импорт рвёт цикл result_checker→sstats_history→
    # prob_calibration→result_checker.
    from .sstats_history import fetch_finished_matches_sstats

    # football-data free отклоняет диапазон шире ~10 дней (400) — капаем;
    # широкое окно catch-up обслуживает sstats.
    fd_days = min(days_back, 10)
    finished = list(await fetch_finished_matches(days_back=fd_days))
    finished += await fetch_finished_matches_sstats(
        days_back=days_back, max_pages=sstats_max_pages)
    if not finished:
        return

    with SessionLocal() as session:
        # Сохраняем свежие результаты в БД. seen_keys ловит дубли В ПРЕДЕЛАХ
        # фетча (два матча с одинаковым нормализованным ключом, перекрытие
        # football-data ↔ sstats, повтор страниц) — session.get их не видит,
        # пока сессия не сфлашена → иначе UNIQUE constraint на батче.
        seen_keys: set[str] = set()
        for m in finished:
            key = _result_key(m.home_team, m.away_team, m.utc_date)
            if key in seen_keys:
                continue
            seen_keys.add(key)
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
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
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
                age_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - first.commence_time.replace(tzinfo=None)
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
                backed = extract_bet_side(a.detector, a.payload or {})
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
