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
    # Elo=1500/1500 → модель ≈ home/away ~0.35, draw ~0.30 (норм.).
    # Рынок сильно смещён: p_home=0.20 → gap=|0.20-0.35|/0.35 ≈ 43% >> 20%
    market_probs = {"home": 0.20, "draw": 0.25, "away": 0.55}
    hits = detect_model_gap(session, match, market_probs, xg_pred=None)
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
    # xG-модель: home=0.50, draw=0.25, away=0.25 (уже сумма 1).
    # Рынок: home=0.20 → gap=|0.20-0.50|/0.50 = 60% → срабатывает.
    pred = XgPrediction(
        home_xg=1.5, away_xg=1.0,
        home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
        home_glicko=1550.0, away_glicko=1500.0,
    )
    market_probs = {"home": 0.20, "draw": 0.25, "away": 0.55}
    hits = detect_model_gap(session, match, market_probs, xg_pred=pred)
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
    """payload['fair'] = 1/p_model; gap считается в простр. вероятностей."""
    match = _match()
    pred = XgPrediction(
        home_xg=2.0, away_xg=0.5,
        home_win_prob=0.625, draw_prob=0.20, away_win_prob=0.175,
        home_glicko=1700.0, away_glicko=1400.0,
    )
    # Рынок точно равен модели → gap = 0 → ничего (threshold = 0.20)
    market_probs = {"home": 0.625, "draw": 0.20, "away": 0.175}
    hits = detect_model_gap(session, match, market_probs, xg_pred=pred)
    assert hits == [], "Когда market == model, gap=0, не срабатывает"

    # Сдвинем p_market_home на 28% ниже модели → должно сработать
    market_probs["home"] = 0.45  # |0.45-0.625|/0.625 ≈ 0.28 ≥ 0.20
    hits = detect_model_gap(session, match, market_probs, xg_pred=pred)
    home_hits = [h for h in hits if h.payload["outcome"] == "home"]
    assert home_hits, "28% gap должен сработать"
    assert home_hits[0].payload["fair"] == pytest.approx(1.60, abs=0.01)
    assert home_hits[0].payload["model_prob"] == pytest.approx(0.625, abs=1e-6)


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
    market_probs = {"home": 0.33, "draw": 0.33, "away": 0.34}
    # Не должно крашнуться: p_model_home=0 → исход home пропускается
    hits = detect_model_gap(session, match, market_probs, xg_pred=pred)
    # Результат не важен, важно что нет ZeroDivisionError
    assert isinstance(hits, list)
