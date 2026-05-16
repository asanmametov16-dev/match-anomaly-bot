"""sstats.net клиент для обогащения детектора model_gap данными xG / winProb / Glicko.

Single-purpose: для каждого MatchOdds находит соответствующий матч в sstats
(по нормализованным именам команд и дате) и подтягивает прогноз. Все ошибки
swallow'аются — на любой fail возвращаем None / пустой dict, и caller
(pipeline) спокойно работает по старой логике через Elo.

Кэш:
- daily index: date_str → {(home_norm, away_norm): sstats_id}  (TTL 24h)
- xG predictions: sstats_id → XgPrediction | None              (TTL 24h)

Лимиты sstats — 150 req/min с ключом. Типичный цикл: 1-2 fetch индекса (день
+ след. день, если кэш холодный) + N fetch'ей xG (по числу матчей). Для 10-15
матчей за цикл это <20 запросов — ниже лимита на порядок.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx

from .config import settings
from .results_client import normalize_team_name

_APIKEY_RE = re.compile(r"apikey=[^&\s'\"]+")


def _redact(msg: object) -> str:
    """Вырезать apikey из строки (URL в исключениях httpx и т.п.)."""
    return _APIKEY_RE.sub("apikey=***", str(msg))

log = logging.getLogger(__name__)

BASE_URL = "https://api.sstats.net"
CACHE_TTL = timedelta(hours=24)
HTTP_TIMEOUT = 30.0


@dataclass
class XgPrediction:
    """Прогноз sstats для одного матча. None-поля не хранятся — если хоть
    одно из xG/winProb отсутствует, мы возвращаем None вместо объекта."""
    home_xg: float
    away_xg: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_glicko: float
    away_glicko: float
    league: str | None = None  # для лиго-зависимого доверия model_gap (#3)


@dataclass
class _IndexEntry:
    index: dict[tuple[str, str], int]
    fetched_at: datetime


def _extract_league(g: dict) -> str | None:
    """«Страна — Лига» из элемента /Games/list (схема season.league)."""
    sl = ((g.get("season") or {}).get("league") or {})
    name = sl.get("name") or (g.get("league") or {}).get("name")
    if not name:
        return None
    country = (sl.get("country") or {}).get("name")
    return f"{country} — {name}" if country else name


# Модуль-уровень кэш — переживает между циклами в одном процессе.
_daily_indexes: dict[str, _IndexEntry] = {}
_xg_cache: dict[int, tuple[datetime, XgPrediction | None]] = {}
_game_league: dict[int, str] = {}  # sstats_id → "Страна — Лига"


def _is_enabled() -> bool:
    return bool(settings.sstats_enabled) and bool(settings.sstats_api_key)


def _fresh(fetched_at: datetime) -> bool:
    return datetime.now(timezone.utc) - fetched_at < CACHE_TTL


async def _fetch_day_index(client: httpx.AsyncClient, date_str: str) -> dict[tuple[str, str], int]:
    """Скачивает /Games/list?Date=YYYY-MM-DD и строит индекс по нормализованным
    именам. При коллизии (двое матчей с одинаковой парой команд за день) —
    оставляем первый. Кэширует на 24ч.
    """
    cached = _daily_indexes.get(date_str)
    if cached and _fresh(cached.fetched_at):
        return cached.index

    try:
        r = await client.get(
            f"{BASE_URL}/Games/list",
            params={"Date": date_str, "Limit": 1000, "apikey": settings.sstats_api_key},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        games = (r.json() or {}).get("data") or []
    except Exception as e:
        log.warning("sstats /Games/list?Date=%s упал: %s", date_str, _redact(e))
        return {}

    index: dict[tuple[str, str], int] = {}
    for g in games:
        ht = ((g.get("homeTeam") or {}).get("name") or "").strip()
        at = ((g.get("awayTeam") or {}).get("name") or "").strip()
        gid = g.get("id")
        if not (ht and at and gid):
            continue
        key = (normalize_team_name(ht), normalize_team_name(at))
        if index.setdefault(key, gid) == gid:  # новая запись (не коллизия)
            lg = _extract_league(g)
            if lg:
                _game_league[gid] = lg

    _daily_indexes[date_str] = _IndexEntry(index=index, fetched_at=datetime.now(timezone.utc))
    log.info("sstats: индекс на %s — %d матчей", date_str, len(index))
    return index


async def _fetch_xg(client: httpx.AsyncClient, sstats_id: int) -> XgPrediction | None:
    """GET /Games/glicko/{id} → XgPrediction. Возвращает None, если у матча
    нет xG/winProb (бывает у второго эшелона / женских / резервов)."""
    cached = _xg_cache.get(sstats_id)
    if cached and _fresh(cached[0]):
        return cached[1]

    try:
        r = await client.get(
            f"{BASE_URL}/Games/glicko/{sstats_id}",
            params={"apikey": settings.sstats_api_key},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        gl = ((r.json() or {}).get("data") or {}).get("glicko") or {}
    except Exception as e:
        log.warning("sstats /Games/glicko/%s упал: %s", sstats_id, _redact(e))
        _xg_cache[sstats_id] = (datetime.now(timezone.utc), None)
        return None

    home_xg = gl.get("homeXg")
    away_xg = gl.get("awayXg")
    hwp = gl.get("homeWinProbability")
    awp = gl.get("awayWinProbability")
    if any(v is None for v in (home_xg, away_xg, hwp, awp)):
        _xg_cache[sstats_id] = (datetime.now(timezone.utc), None)
        return None

    dwp = max(0.01, 1.0 - float(hwp) - float(awp))
    pred = XgPrediction(
        home_xg=float(home_xg),
        away_xg=float(away_xg),
        home_win_prob=float(hwp),
        draw_prob=dwp,
        away_win_prob=float(awp),
        home_glicko=float(gl.get("homeRating") or 0.0),
        away_glicko=float(gl.get("awayRating") or 0.0),
    )
    _xg_cache[sstats_id] = (datetime.now(timezone.utc), pred)
    return pred


async def fetch_xg_batch(matches: Iterable) -> dict[str, XgPrediction]:
    """Главный API для pipeline: вернёт {match_id → XgPrediction} для матчей,
    у которых нашёлся sstats-аналог с xG. Матчи без покрытия просто отсутствуют
    в результате. Никогда не кидает исключений.
    """
    if not _is_enabled():
        return {}

    out: dict[str, XgPrediction] = {}
    matches_list = list(matches)
    if not matches_list:
        return out

    needed_dates = sorted({
        m.commence_time.astimezone(timezone.utc).date().isoformat()
        for m in matches_list
    })

    try:
        async with httpx.AsyncClient() as client:
            # 1) кэшируем daily-индексы для всех нужных дат
            for date_str in needed_dates:
                await _fetch_day_index(client, date_str)

            # 2) для каждого матча — lookup в индексе, затем GET /Games/glicko
            for m in matches_list:
                date_str = m.commence_time.astimezone(timezone.utc).date().isoformat()
                cached_idx = _daily_indexes.get(date_str)
                if cached_idx is None:
                    continue
                home_n = normalize_team_name(m.home_team)
                away_n = normalize_team_name(m.away_team)
                sstats_id = cached_idx.index.get((home_n, away_n))
                if sstats_id is None:
                    log.debug("sstats: не нашёл %s vs %s на %s", m.home_team, m.away_team, date_str)
                    continue
                pred = await _fetch_xg(client, sstats_id)
                if pred is not None:
                    pred.league = _game_league.get(sstats_id)
                    out[m.match_id] = pred
    except Exception as e:
        # Полная резервная защита — не должна срабатывать (вложенные блоки уже ловят),
        # но если что-то упадёт на уровне AsyncClient/сети — просто отдадим что собрали.
        log.warning("sstats batch fetch упал: %s", _redact(e))

    log.info("sstats xG-обогащение: %d / %d матчей покрыты", len(out), len(matches_list))
    return out
