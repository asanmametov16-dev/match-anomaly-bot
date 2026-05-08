"""Клиент football-data.org для получения результатов сыгранных матчей.

Бесплатный тариф: 10 запросов в минуту, доступ к топ-лигам (PL, La Liga,
Bundesliga, Serie A, Ligue 1, Champions League и т.д.). Регистрация:
https://www.football-data.org/client/register

Используется ежедневным джобом для обновления Elo-рейтингов.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"


@dataclass
class FinishedMatch:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    utc_date: datetime
    competition: str


async def fetch_finished_matches(days_back: int = 2) -> list[FinishedMatch]:
    """Забирает результаты матчей за последние `days_back` дней.

    Если ключ не задан, возвращает пустой список — это нормально, можно
    запускать систему и без обновления Elo (детекторы spread/drift работают
    независимо).
    """
    if not settings.football_data_key:
        log.info("FOOTBALL_DATA_KEY не задан — пропускаю обновление Elo")
        return []

    date_to = date.today()
    date_from = date_to - timedelta(days=days_back)

    url = f"{BASE_URL}/matches"
    params = {
        "dateFrom": date_from.isoformat(),
        "dateTo": date_to.isoformat(),
        "status": "FINISHED",
    }
    headers = {"X-Auth-Token": settings.football_data_key}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as e:
            log.error("football-data.org: ошибка запроса: %s", e)
            return []
        data = response.json()

    matches: list[FinishedMatch] = []
    for raw in data.get("matches", []):
        score = raw.get("score", {}).get("fullTime", {})
        home = score.get("home")
        away = score.get("away")
        if home is None or away is None:
            continue
        matches.append(FinishedMatch(
            home_team=raw["homeTeam"]["name"],
            away_team=raw["awayTeam"]["name"],
            home_score=int(home),
            away_score=int(away),
            utc_date=datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            competition=raw.get("competition", {}).get("name", ""),
        ))

    log.info("football-data.org: получено %d завершённых матчей за последние %d дн.",
             len(matches), days_back)
    return matches


# --- Сопоставление имён команд ---------------------------------------------
# The Odds API и football-data.org называют команды по-разному.
# Порядок обработки: акценты → пунктуация → lowercase → числа →
# суффиксы/префиксы FC/SC/CA/Club → словарь алиасов.

# Аббревиатуры, убираемые с конца названия ("Arsenal FC" → "Arsenal")
_FC_SUFFIXES = frozenset({
    "fc", "cf", "afc", "sc", "bc", "ac", "fk", "sk", "bk",
    "if", "ik", "sv", "tsv", "vfb", "vfl", "rgk",
    # Бразильские аббревиатуры штатов: "Palmeiras-SP" → "Palmeiras"
    "sp", "rj", "mg", "rs", "pr", "ba", "ce", "pe",
    # Корпоративные суффиксы: "Botafogo FR" → "Botafogo"
    "fr", "cr",
})

# Организационные префиксы, убираемые с начала ("CA Rosario" → "Rosario",
# "FC Bayern" → "Bayern", "Club Atlético" → "Atlético").
# Отдельный frozenset, т.к. часть слов безопасна как суффикс, но не как префикс,
# и наоборот (например, "if" убираем только с конца).
_ORG_PREFIXES = frozenset({
    "fc", "cf", "afc", "sc", "bc", "ac", "fk", "sk", "bk",
    "sv", "tsv",
    "ca", "cd", "cs", "as", "se", "ec", "rc", "us", "ss",
    "club", "clube",   # "Club Atlético" / "Clube Atlético Mineiro"
    "deportivo",       # "Deportivo Riestra" → этот матч уже распознан
})

# Известные расхождения между Odds API и football-data.org
_ALIASES: dict[str, str] = {
    # Европейские клубы
    "internazionale":            "inter milan",
    "paris sg":                  "paris saint germain",
    "paris saint-germain":       "paris saint germain",
    "wolverhampton wanderers":   "wolverhampton",
    "newcastle united":          "newcastle",
    "brighton hove albion":      "brighton",
    "nottingham forest":         "nott'm forest",
    "borussia gladbach":         "borussia monchengladbach",
    "borussia m gladbach":       "borussia monchengladbach",
    "rb leipzig":                "rasenballsport leipzig",
    "atletico de madrid":        "atletico madrid",
    "real sociedad":             "sociedad",
    # Немецкие клубы (города: немецкое ↔ английское написание)
    "bayern munchen":            "bayern munich",
    "munchen":                   "munich",
    # Копа Либертадорес / Южная Америка
    "estudiantes de la plata":   "estudiantes la plata",
    "atletico mineiro":          "mineiro",       # "CA Mineiro" → "mineiro"
    "libertad asuncion":         "libertad",
    "olimpia asuncion":          "olimpia",
    "cristal":                   "sporting cristal",  # "CS Cristal" → "cristal"
    "palmeiras":                 "palmeiras",     # нет изменений, но явно
    # Японские клубы (football-data может добавлять/убирать "FC")
    "gamba osaka":               "gamba osaka",
}


def normalize_team_name(name: str) -> str:
    """Нормализация имени команды для сопоставления между двумя API.

    Шаги: акценты → пунктуация → lowercase → числа →
    суффиксы → префиксы → алиасы.
    """
    # Убираем акценты: é→e, ü→u, ñ→n и т.д.
    n = unicodedata.normalize("NFKD", name.strip())
    n = "".join(c for c in n if not unicodedata.combining(c))
    # Убираем пунктуацию (точки, апострофы и т.п.), оставляем буквы/цифры/пробелы
    n = re.sub(r"[^\w\s]", " ", n)
    # Lowercase и схлопываем пробелы
    n = re.sub(r"\s+", " ", n.lower().strip())
    # Убираем отдельно стоящие числа ("Bayer 04 Leverkusen" → "Bayer Leverkusen")
    n = re.sub(r"\b\d+\b", "", n)
    n = re.sub(r"\s+", " ", n.strip())
    words = n.split()
    # Убираем организационные суффиксы с конца ("Arsenal FC" → "Arsenal")
    while words and words[-1] in _FC_SUFFIXES:
        words.pop()
    # Убираем организационные префиксы с начала ("CA Rosario" → "Rosario",
    # "FC Bayern" → "Bayern", "Club Atlético de Madrid" → "Atlético de Madrid")
    while words and words[0] in _ORG_PREFIXES:
        words.pop(0)
    n = " ".join(words)
    # Применяем словарь алиасов
    return _ALIASES.get(n, n)
