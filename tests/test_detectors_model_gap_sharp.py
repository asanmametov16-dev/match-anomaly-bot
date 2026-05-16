"""Тесты sharp-консенсус fallback для detect_model_gap (тир между xG и Elo).

Синтетика, in-memory SQLite только для Elo-пути (get_rating).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base
from src.detectors import detect_model_gap
from src.odds_client import BookmakerOdds, MatchOdds
from src.probability import sharp_consensus_probabilities
from src.sstats_client import XgPrediction


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        yield s


def _match(bms: list[BookmakerOdds]) -> MatchOdds:
    return MatchOdds(
        match_id="m1", sport_key="soccer_test",
        home_team="TeamA", away_team="TeamB",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=48),
        bookmakers=bms,
    )


def _bm(name, h, d, a):
    return BookmakerOdds(bookmaker=name, home=h, draw=d, away=a)


# --- helper: sharp_consensus_probabilities ----------------------------------

def test_helper_none_below_min_books():
    m = _match([_bm("pinnacle", 2.0, 3.6, 4.0), _bm("bk_soft", 2.1, 3.5, 3.9)])
    assert sharp_consensus_probabilities(m, min_books=2) is None  # 1 sharp < 2


def test_helper_ignores_soft_books():
    """Дикая soft-котировка не должна сдвигать sharp-консенсус."""
    m = _match([
        _bm("pinnacle", 2.00, 3.60, 4.00),
        _bm("betfair_ex_eu", 2.00, 3.60, 4.00),
        _bm("bk_soft", 10.0, 1.20, 30.0),  # абсурд — должен игнорироваться
    ])
    probs = sharp_consensus_probabilities(m, min_books=2)
    assert probs is not None
    assert probs["home"] == pytest.approx(probs["away"], abs=0.12) or probs["home"] > 0.45
    # home (коэф 2.0) — фаворит, ~0.5, не задран soft-выбросом
    assert 0.45 < probs["home"] < 0.55
    assert sum(probs.values()) == pytest.approx(1.0, abs=0.05)


# --- detect_model_gap: тир sharp_consensus ----------------------------------

def test_uses_sharp_consensus_when_no_xg(session):
    m = _match([
        _bm("pinnacle", 2.00, 3.60, 4.00),
        _bm("betfair_ex_eu", 2.00, 3.60, 4.00),
        _bm("bk_soft", 2.05, 3.55, 3.95),
    ])
    # sharp fair_home ≈ 2.0; рынок 4.0 → gap ≈ 100% >> 20%
    medians = {"home": 4.0, "draw": 3.6, "away": 2.0}
    hits = detect_model_gap(session, m, medians, xg_pred=None)
    assert hits
    for h in hits:
        assert h.payload["source"] == "sharp_consensus"
        assert "sharp_p_home" in h.payload
        assert "rating_home" not in h.payload
    home = [h for h in hits if h.payload["outcome"] == "home"][0]
    assert home.payload["fair"] == pytest.approx(1.0 / home.payload["sharp_p_home"],
                                                 rel=1e-6)


def test_falls_back_to_elo_without_sharp_books(session):
    m = _match([_bm("bk1", 4.5, 3.5, 1.8), _bm("bk2", 4.4, 3.6, 1.82)])
    medians = {"home": 4.5, "draw": 3.5, "away": 1.8}
    hits = detect_model_gap(session, m, medians, xg_pred=None)
    assert hits, "Без sharp-контор ожидаем Elo-путь со срабатыванием"
    for h in hits:
        assert h.payload["source"] == "elo"
        assert "rating_home" in h.payload
        assert "sharp_p_home" not in h.payload


def test_xg_takes_priority_over_sharp(session):
    m = _match([
        _bm("pinnacle", 2.00, 3.60, 4.00),
        _bm("betfair_ex_eu", 2.00, 3.60, 4.00),
    ])
    pred = XgPrediction(
        home_xg=1.5, away_xg=1.0,
        home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
        home_glicko=1550.0, away_glicko=1500.0,
    )
    medians = {"home": 4.0, "draw": 3.5, "away": 2.0}
    hits = detect_model_gap(session, m, medians, xg_pred=pred)
    assert hits
    for h in hits:
        assert h.payload["source"] == "sstats_xg"
        assert "sharp_p_home" not in h.payload
