"""Тесты для detect_model_gap — путь с xG-обогащением и fallback на Elo.

Все тесты — синтетические, без сетевых вызовов и без реальной БД (in-memory
SQLite поднимается локально для get_rating).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base
from src.detectors import detect_model_gap
from src.odds_client import BookmakerOdds, MatchOdds
from src.sstats_client import XgPrediction


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        yield s


def _match() -> MatchOdds:
    return MatchOdds(
        match_id="m1",
        sport_key="soccer_test",
        home_team="TeamA",
        away_team="TeamB",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=48),
        bookmakers=[BookmakerOdds(bookmaker="bm", home=2.0, draw=3.5, away=4.0)],
    )


# ---------------------------------------------------------------------------
# Fallback: без xg_pred работает старым способом через Elo
# ---------------------------------------------------------------------------

def test_falls_back_to_elo_when_no_xg(session):
    match = _match()
    # С Elo=1500 у обоих fair_home ≈ 2.38. Сильное отклонение market'а:
    # home=4.5 → gap = |4.5-2.38|/2.38 ≈ 89% >> threshold 20%
    medians = {"home": 4.5, "draw": 3.5, "away": 1.8}
    hits = detect_model_gap(session, match, medians, xg_pred=None)
    assert hits, "Без xG ожидаем срабатывание на дефолтных Elo при сильном отклонении"
    for h in hits:
        assert h.payload["source"] == "elo"
        assert "rating_home" in h.payload
        assert "home_xg" not in h.payload


# ---------------------------------------------------------------------------
# С xG: fair_odds из winProb, source="sstats_xg"
# ---------------------------------------------------------------------------

def test_uses_sstats_xg_when_provided(session):
    match = _match()
    # xG-модель говорит home_win=0.50, away=0.25, draw=0.25
    # → fair_home=2.0, fair_draw=4.0, fair_away=4.0
    # market: home=4.0, draw=3.5, away=2.0
    # gap на home: |4.0-2.0|/2.0 = 100% → srабатывает
    # gap на away: |2.0-4.0|/4.0 = 50% → срабатывает
    pred = XgPrediction(
        home_xg=1.5, away_xg=1.0,
        home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
        home_glicko=1550.0, away_glicko=1500.0,
    )
    medians = {"home": 4.0, "draw": 3.5, "away": 2.0}
    hits = detect_model_gap(session, match, medians, xg_pred=pred)
    assert hits, "С xG-предсказанием должны быть срабатывания"
    for h in hits:
        assert h.payload["source"] == "sstats_xg"
        assert h.payload["home_xg"] == pytest.approx(1.5)
        assert h.payload["home_glicko"] == pytest.approx(1550.0)
        assert "rating_home" not in h.payload


# ---------------------------------------------------------------------------
# Корректное вычисление fair_odds из winProb
# ---------------------------------------------------------------------------

def test_xg_fair_odds_math(session):
    """fair_home должен быть 1/home_win_prob, fair_away 1/away_win_prob."""
    match = _match()
    pred = XgPrediction(
        home_xg=2.0, away_xg=0.5,
        home_win_prob=0.625, draw_prob=0.20, away_win_prob=0.175,
        home_glicko=1700.0, away_glicko=1400.0,
    )
    # Market точно равен fair → gap = 0 → ничего не сработает (threshold = 0.20)
    medians = {"home": 1.60, "draw": 5.0, "away": 5.714}  # = 1/0.625, 1/0.20, 1/0.175
    hits = detect_model_gap(session, match, medians, xg_pred=pred)
    assert hits == [], "Когда market == fair, gap=0, не срабатывает"

    # Сдвинем market home на 30% выше fair → должно сработать
    medians["home"] = 1.60 * 1.30  # = 2.08
    hits = detect_model_gap(session, match, medians, xg_pred=pred)
    home_hits = [h for h in hits if h.payload["outcome"] == "home"]
    assert home_hits, "30% gap должен сработать"
    assert home_hits[0].payload["fair"] == pytest.approx(1.60, abs=0.01)


# ---------------------------------------------------------------------------
# Граничный кейс: winProb=0 → защита от деления на ноль
# ---------------------------------------------------------------------------

def test_xg_zero_probability_handled(session):
    """Если sstats вернёт 0 win-prob — fair_odds считаем через clamp 0.01."""
    match = _match()
    pred = XgPrediction(
        home_xg=0.0, away_xg=0.0,
        home_win_prob=0.0, draw_prob=0.0, away_win_prob=1.0,  # вырожденный случай
        home_glicko=1500, away_glicko=1500,
    )
    medians = {"home": 5.0, "draw": 5.0, "away": 5.0}
    # Не должно крашнуться (clamp 0.01 → fair max 100.0)
    hits = detect_model_gap(session, match, medians, xg_pred=pred)
    # Результат не важен, важно что нет ZeroDivisionError
    assert isinstance(hits, list)
