"""Клиент для The Odds API (https://the-odds-api.com).

Возвращает список матчей с коэффициентами от разных букмекеров.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


@dataclass
class BookmakerOdds:
    bookmaker: str
    home: float | None
    draw: float | None
    away: float | None
    # Дополнительные рынки. Каждый — список словарей outcome→price вместе
    # с точкой (line). Пример для totals:
    # [{"name": "Over", "price": 1.85, "point": 2.5}, {"name": "Under", ...}]
    totals: list[dict] | None = None
    spreads: list[dict] | None = None


@dataclass
class MatchOdds:
    match_id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmakers: list[BookmakerOdds]


def _parse_match(raw: dict[str, Any]) -> MatchOdds | None:
    """Конвертирует JSON от API в MatchOdds. Возвращает None, если нет h2h-рынков."""
    bookmakers: list[BookmakerOdds] = []
    home_team = raw["home_team"]
    away_team = raw["away_team"]

    for bm in raw.get("bookmakers", []):
        h2h_outcomes: dict[str, float] = {}
        totals: list[dict] = []
        spreads: list[dict] = []

        for market in bm.get("markets", []):
            key = market["key"]
            if key == "h2h":
                h2h_outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            elif key == "totals":
                totals = [
                    {"name": o["name"], "price": o["price"], "point": o.get("point")}
                    for o in market.get("outcomes", [])
                ]
            elif key == "spreads":
                spreads = [
                    {"name": o["name"], "price": o["price"], "point": o.get("point")}
                    for o in market.get("outcomes", [])
                ]

        if not h2h_outcomes:
            continue

        bookmakers.append(BookmakerOdds(
            bookmaker=bm["key"],
            home=h2h_outcomes.get(home_team),
            draw=h2h_outcomes.get("Draw"),
            away=h2h_outcomes.get(away_team),
            totals=totals or None,
            spreads=spreads or None,
        ))

    if not bookmakers:
        return None

    return MatchOdds(
        match_id=raw["id"],
        sport_key=raw["sport_key"],
        home_team=home_team,
        away_team=away_team,
        commence_time=datetime.fromisoformat(raw["commence_time"].replace("Z", "+00:00")),
        bookmakers=bookmakers,
    )


async def fetch_odds() -> list[MatchOdds]:
    """Забирает текущие коэффициенты по настроенному виду спорта."""
    url = f"{BASE_URL}/sports/{settings.odds_api_sport}/odds"
    params = {
        "apiKey": settings.odds_api_key,
        "regions": settings.odds_api_regions,
        "markets": settings.odds_api_markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    # API возвращает в заголовках, сколько запросов осталось — полезно логировать
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    log.info("Odds API: получено %d матчей, осталось запросов: %s, использовано: %s",
             len(data), remaining, used)

    matches = [m for raw in data if (m := _parse_match(raw)) is not None]
    return matches
