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
from datetime import datetime, timedelta, timezone

import httpx

from .config import settings
from .db import SessionLocal, SstatsModelOutcome, init_db
from .prob_calibration import brier_score, log_loss
from .results_client import FinishedMatch
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
                   sleep: float = _RATE_SLEEP, start_offset: int = 0,
                   max_pages: int | None = None,
                   stop_on_known_page: bool = True) -> int:
    """Подтянуть до max_games исторических матчей с winProb. Идемпотентно
    (game_id — PK, уже записанные пропускаются). Возвращает число новых.

    Глубокая выборка: при stop_on_known_page=False цикл НЕ останавливается на
    полностью известной странице (нужно, т.к. Offset=0 = свежие = уже в БД),
    а идёт дальше по offset до пустой страницы / max_pages / max_games. С
    start_offset можно продолжить с известной глубины.
    """
    if not (settings.sstats_api_key and settings.sstats_enabled):
        log.error("SSTATS_API_KEY не задан / sstats выключен — бэкфилл невозможен")
        return 0

    init_db()
    written = 0
    offset = start_offset
    pages = 0

    async with httpx.AsyncClient() as client:
        while written < max_games:
            if max_pages is not None and pages >= max_pages:
                break
            page = await _fetch_ended_page(client, offset, page_limit)
            if not page:
                break  # данные кончились — настоящий конец истории
            offset += page_limit
            pages += 1
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

            if page_new == 0 and stop_on_known_page:
                # мелкий режим: первая же известная страница = конец свежих
                break

    log.info("sstats backfill: записано %d матчей", written)
    return written


async def fetch_finished_matches_sstats(
    days_back: int = 3, max_pages: int = 8
) -> list[FinishedMatch]:
    """Свежие сыгранные матчи из sstats как ВТОРОЙ источник результатов.

    Покрывает лиги вне football-data.org (MLS, Süper Lig, Eredivisie и т.д.),
    которые иначе никогда не резолвятся → нет CLV/Brier/outcomes. Только
    /Games/list (без per-game glicko) — дёшево. Order=-1 (свежие первыми):
    как только встретили матч старше окна — дальше только старее, стоп.
    Никогда не кидает исключений (как остальной sstats-код).
    """
    if not (settings.sstats_api_key and settings.sstats_enabled):
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)) \
        .replace(tzinfo=None)
    out: list[FinishedMatch] = []

    try:
        async with httpx.AsyncClient() as client:
            offset = 0
            for _ in range(max_pages):
                page = await _fetch_ended_page(client, offset, 200)
                if not page:
                    break
                offset += 200
                reached_old = False
                for item in page:
                    p = _parse_ended_item(item)
                    if p is None or p["played"] is None:
                        continue  # без даты не построить result_key — пропуск
                    if p["played"] < cutoff:
                        reached_old = True
                        continue
                    out.append(FinishedMatch(
                        home_team=p["home"], away_team=p["away"],
                        home_score=p["hs"], away_score=p["as"],
                        utc_date=p["played"], competition=p["league"] or "",
                    ))
                if reached_old:
                    break  # Order=-1 → глубже только ещё старее
    except Exception as e:
        log.warning("sstats fetch_finished упал: %s", e)

    log.info("sstats: второй источник результатов — %d матчей", len(out))
    return out


# --- Лиго-зависимое доверие модели (#3) -------------------------------------

# Горячий кэш: league → "trusted" | "unreliable". Отсутствие = "unknown"
# (мало истории или Brier в нейтральной зоне) — не гейтим.
_trust: dict[str, str] = {}
_UNIFORM_BRIER = 2.0 / 3.0


def refresh_model_trust() -> dict[str, str]:
    """Пересчитать доверие к модели по лигам из SstatsModelOutcome.

    trusted   = Brier заметно ниже равномерного (модель информативна);
    unreliable = Brier ≥ равномерного (модель не лучше монетки);
    между / мало данных = не в кэше → "unknown" (нейтрально, не гейтим).
    Вызывается шедулером ежечасно и на старте.
    """
    from sqlalchemy import func, select

    margin = settings.model_trust_uniform_margin
    min_n = settings.model_trust_min_samples
    new: dict[str, str] = {}

    with SessionLocal() as session:
        rows = session.execute(
            select(
                SstatsModelOutcome.league,
                func.count(SstatsModelOutcome.game_id),
                func.avg(SstatsModelOutcome.brier),
            ).group_by(SstatsModelOutcome.league)
        ).all()

    for league, n, brier in rows:
        if not league or int(n or 0) < min_n or brier is None:
            continue
        b = float(brier)
        if b < _UNIFORM_BRIER - margin:
            new[league] = "trusted"
        elif b >= _UNIFORM_BRIER:
            new[league] = "unreliable"
        # нейтральная зона → не сохраняем (unknown)

    _trust.clear()
    _trust.update(new)
    log.info("model_trust: %d лиг классифицировано (%d trusted, %d unreliable)",
             len(new), sum(v == "trusted" for v in new.values()),
             sum(v == "unreliable" for v in new.values()))
    return dict(_trust)


def league_model_trust(league: str | None) -> str:
    """Доверие к sstats-модели в лиге: trusted | unreliable | unknown."""
    if not league:
        return "unknown"
    return _trust.get(league, "unknown")


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
