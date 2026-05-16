"""Офлайн-бэкфилл исторической калибровки модели sstats.

`/Games/list?Ended=true` отдаёт сыгранные матчи (счёт, лига, дата), а
`/Games/glicko/{id}` — winProb/xG модели. Скорим прогноз модели против
факта тем же Brier/log-loss, что и рыночную калибровку (переиспользуем
prob_calibration). Запускается скриптом scripts/backfill_sstats_history.py,
НЕ в пайплайне (сетевой батч с rate-limit).

Парсер /Games/list терпим к схеме: точные имена полей счёта/даты/лиги в
sstats заранее не зафиксированы — пробуем набор кандидатов и пропускаем
запись, если ключевого поля нет (как в остальном sstats-коде).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from .config import settings
from .db import SessionLocal, SstatsModelOutcome, init_db
from .prob_calibration import brier_score, log_loss
from .sstats_client import BASE_URL, _fetch_xg

log = logging.getLogger(__name__)

_RATE_SLEEP = 0.45  # ~133 glicko-запросов/мин < лимита sstats 150/мин


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _parse_ended_item(g: dict) -> dict | None:
    """Достаёт id/лигу/дату/команды/счёт из элемента /Games/list?Ended.

    Реальная схема sstats: счёт — homeFTResult/awayFTResult (фолбэк на
    homeResult/awayResult), лига — season.league.name (+ страна), дата —
    date (ISO). Остальные кандидаты оставлены как страховка.
    """
    gid = g.get("id")
    home = ((g.get("homeTeam") or {}).get("name") or "").strip()
    away = ((g.get("awayTeam") or {}).get("name") or "").strip()

    hs = _first(g, "homeFTResult", "homeResult", "homeScore", "scoreHome")
    if hs is None:
        hs = _first((g.get("score") or {}), "home", "fullTimeHome")
    as_ = _first(g, "awayFTResult", "awayResult", "awayScore", "scoreAway")
    if as_ is None:
        as_ = _first((g.get("score") or {}), "away", "fullTimeAway")

    if not (gid and home and away) or hs is None or as_ is None:
        return None

    season_league = ((g.get("season") or {}).get("league") or {})
    league = season_league.get("name") \
        or _first(g, "leagueName") \
        or (g.get("league") or {}).get("name")
    country = (season_league.get("country") or {}).get("name")
    if league and country:
        league = f"{country} — {league}"

    played = None
    raw_dt = _first(g, "date", "dateUtc", "startDate", "utcDate")
    if isinstance(raw_dt, str):
        try:
            played = datetime.fromisoformat(raw_dt.replace("Z", "+00:00")) \
                .replace(tzinfo=None)
        except ValueError:
            played = None

    try:
        hs, as_ = int(hs), int(as_)
    except (TypeError, ValueError):
        return None

    return {"game_id": int(gid), "league": league, "played": played,
            "home": home, "away": away, "hs": hs, "as": as_}


def _actual(hs: int, as_: int) -> str:
    if hs > as_:
        return "home"
    if as_ > hs:
        return "away"
    return "draw"


async def _fetch_ended_page(client: httpx.AsyncClient, offset: int,
                            limit: int) -> list[dict]:
    try:
        r = await client.get(
            f"{BASE_URL}/Games/list",
            params={"Ended": "true", "Order": -1, "Limit": limit,
                    "Offset": offset, "apikey": settings.sstats_api_key},
            timeout=30.0,
        )
        r.raise_for_status()
        return (r.json() or {}).get("data") or []
    except Exception as e:
        log.warning("sstats /Games/list?Ended Offset=%s упал: %s", offset, e)
        return []


async def backfill(max_games: int = 1500, page_limit: int = 200,
                   sleep: float = _RATE_SLEEP) -> int:
    """Подтянуть до max_games исторических матчей с winProb. Идемпотентно
    (game_id — PK, уже записанные пропускаются). Возвращает число новых.
    """
    if not (settings.sstats_api_key and settings.sstats_enabled):
        log.error("SSTATS_API_KEY не задан / sstats выключен — бэкфилл невозможен")
        return 0

    init_db()
    written = 0
    offset = 0

    async with httpx.AsyncClient() as client:
        while written < max_games:
            page = await _fetch_ended_page(client, offset, page_limit)
            if not page:
                break
            offset += page_limit
            page_new = 0

            with SessionLocal() as session:
                for item in page:
                    if written >= max_games:
                        break
                    parsed = _parse_ended_item(item)
                    if parsed is None:
                        continue
                    if session.get(SstatsModelOutcome, parsed["game_id"]):
                        continue

                    pred = await _fetch_xg(client, parsed["game_id"])
                    await asyncio.sleep(sleep)
                    if pred is None:
                        continue  # нет winProb/xG — нечего калибровать

                    probs = {"home": pred.home_win_prob,
                             "draw": pred.draw_prob,
                             "away": pred.away_win_prob}
                    tot = sum(probs.values())
                    if tot <= 0:
                        continue
                    probs = {k: v / tot for k, v in probs.items()}
                    actual = _actual(parsed["hs"], parsed["as"])

                    session.add(SstatsModelOutcome(
                        game_id=parsed["game_id"],
                        league=parsed["league"],
                        played_date=parsed["played"],
                        home_team=parsed["home"], away_team=parsed["away"],
                        p_home=probs["home"], p_draw=probs["draw"],
                        p_away=probs["away"],
                        home_xg=pred.home_xg, away_xg=pred.away_xg,
                        actual=actual,
                        brier=brier_score(probs, actual),
                        log_loss=log_loss(probs, actual),
                    ))
                    written += 1
                    page_new += 1
                session.commit()

            if page_new == 0:
                # страница без новых записей → пагинация исчерпана/не движется
                break

    log.info("sstats backfill: записано %d матчей", written)
    return written


def sstats_model_summary(top_leagues: int = 10) -> dict:
    """Агрегат калибровки модели: overall + по лигам (для /modelcal и #3)."""
    from sqlalchemy import func, select

    with SessionLocal() as session:
        n = session.scalar(select(func.count(SstatsModelOutcome.game_id))) or 0
        if not n:
            return {"n": 0}
        mean_brier = session.scalar(select(func.avg(SstatsModelOutcome.brier)))
        mean_ll = session.scalar(select(func.avg(SstatsModelOutcome.log_loss)))
        per_league = session.execute(
            select(
                SstatsModelOutcome.league,
                func.count(SstatsModelOutcome.game_id),
                func.avg(SstatsModelOutcome.brier),
            )
            .group_by(SstatsModelOutcome.league)
            .order_by(func.count(SstatsModelOutcome.game_id).desc())
            .limit(top_leagues)
        ).all()

    return {
        "n": int(n),
        "mean_brier": float(mean_brier),
        "mean_log_loss": float(mean_ll),
        "uniform_brier": 2.0 / 3.0,
        "leagues": [
            {"league": lg or "—", "n": int(c), "brier": float(b)}
            for lg, c, b in per_league
        ],
    }
