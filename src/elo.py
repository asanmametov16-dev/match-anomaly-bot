"""Elo-рейтинг команд и расчёт 'честного' коэффициента.

На старте рейтинги одинаковые — модель будет накапливать данные постепенно.
Альтернатива: загрузить готовые рейтинги (например, с clubelo.com), но для MVP
оставим простой вариант.

Важно: с пустой базой рейтингов детектор model_gap первое время будет шуметь.
Это нормально — отключи его в .env (поставь MODEL_GAP_THRESHOLD=10) на первые
1–2 недели, пока рейтинги откалибруются по результатам матчей.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from .config import settings
from .db import TeamRating

# Префиксы/суффиксы, которые мешают матчингу команд между источниками данных
_NORMALIZATION_SUFFIXES = (" FC", " CF", " AFC", " SC", " BC")


def _normalize(name: str) -> str:
    n = name.strip()
    for suffix in _NORMALIZATION_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n.lower()


@dataclass
class FairOdds:
    home: float
    draw: float
    away: float


def get_rating(session: Session, team: str) -> float:
    key = _normalize(team)
    row = session.get(TeamRating, key)
    if row is None:
        row = TeamRating(team=key, rating=settings.elo_default_rating, games_played=0)
        session.add(row)
        session.flush()
    return row.rating


def expected_score(rating_a: float, rating_b: float) -> float:
    """Вероятность, что A победит B (без учёта ничьей)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def fair_odds_1x2(rating_home: float, rating_away: float,
                  draw_share: float = 0.26) -> FairOdds:
    """Очень упрощённая модель 1X2.

    Берём вероятность победы по Elo с поправкой на домашнее поле, фиксируем
    'базовую' долю ничьих ~26% (среднее по футболу) и распределяем оставшееся
    между home/away пропорционально Elo-вероятностям.
    """
    p_home_raw = expected_score(
        rating_home + settings.elo_home_advantage, rating_away
    )
    p_away_raw = 1.0 - p_home_raw
    non_draw = 1.0 - draw_share

    p_home = p_home_raw * non_draw
    p_away = p_away_raw * non_draw
    p_draw = draw_share

    # Защита от нулей
    p_home = max(p_home, 0.01)
    p_away = max(p_away, 0.01)
    p_draw = max(p_draw, 0.01)

    return FairOdds(home=1 / p_home, draw=1 / p_draw, away=1 / p_away)


def update_ratings(session: Session, home_team: str, away_team: str,
                   home_score: int, away_score: int) -> None:
    """Обновить рейтинги после сыгранного матча.

    Вызывается из elo_updater.py по результатам с football-data.org.
    """
    home_key = _normalize(home_team)
    away_key = _normalize(away_team)

    home = session.get(TeamRating, home_key)
    if home is None:
        home = TeamRating(team=home_key, rating=settings.elo_default_rating, games_played=0)
        session.add(home)

    away = session.get(TeamRating, away_key)
    if away is None:
        away = TeamRating(team=away_key, rating=settings.elo_default_rating, games_played=0)
        session.add(away)

    if home_score > away_score:
        actual_home, actual_away = 1.0, 0.0
    elif home_score < away_score:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home = actual_away = 0.5

    expected_home = expected_score(
        home.rating + settings.elo_home_advantage, away.rating
    )
    expected_away = 1.0 - expected_home
    k = settings.elo_k_factor

    home.rating += k * (actual_home - expected_home)
    away.rating += k * (actual_away - expected_away)
    home.games_played = (home.games_played or 0) + 1
    away.games_played = (away.games_played or 0) + 1
