"""Тесты лиго-зависимого доверия модели sstats (#3).

refresh_model_trust по SstatsModelOutcome → классификация лиг; проброс
league/model_trust в payload model_gap; демоция в precision-gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.sstats_history as sh
from src.db import Base, SstatsModelOutcome
from src.detectors import AnomalyHit, classify_signal, detect_model_gap
from src.odds_client import BookmakerOdds, MatchOdds
from src.sstats_client import XgPrediction


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(sh, "SessionLocal", Session)
    monkeypatch.setattr(sh.settings, "model_trust_min_samples", 5)
    monkeypatch.setattr(sh.settings, "model_trust_uniform_margin", 0.02)
    yield Session


@pytest.fixture(autouse=True)
def clean_trust():
    sh._trust.clear()
    yield
    sh._trust.clear()


def _rows(s, league, n, brier):
    for i in range(n):
        s.add(SstatsModelOutcome(
            game_id=hash((league, i)) & 0x7FFFFFFF,
            league=league, home_team="H", away_team="A",
            p_home=0.4, p_draw=0.3, p_away=0.3, actual="home",
            brier=brier, log_loss=1.0,
        ))


# --- refresh_model_trust -----------------------------------------------------

def test_classifies_leagues(db):
    with db() as s:
        _rows(s, "Good League", 10, brier=0.50)    # << 0.667 → trusted
        _rows(s, "Bad League", 10, brier=0.75)     # >= 0.667 → unreliable
        _rows(s, "Meh League", 10, brier=0.66)     # нейтрально → unknown
        _rows(s, "Tiny League", 3, brier=0.40)     # мало данных → unknown
        s.commit()

    sh.refresh_model_trust()
    assert sh.league_model_trust("Good League") == "trusted"
    assert sh.league_model_trust("Bad League") == "unreliable"
    assert sh.league_model_trust("Meh League") == "unknown"
    assert sh.league_model_trust("Tiny League") == "unknown"
    assert sh.league_model_trust(None) == "unknown"


# --- проброс league/model_trust в payload model_gap -------------------------

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as s:
        yield s


def _match():
    return MatchOdds(
        match_id="m1", sport_key="soccer_test",
        home_team="A", away_team="B",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=48),
        bookmakers=[BookmakerOdds(bookmaker="bm", home=2.0, draw=3.5, away=4.0)],
    )


def test_model_gap_stamps_league_and_trust(session, monkeypatch):
    monkeypatch.setitem(sh._trust, "England — Premier League", "trusted")
    pred = XgPrediction(
        home_xg=1.5, away_xg=1.0,
        home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
        home_glicko=1550.0, away_glicko=1500.0,
        league="England — Premier League",
    )
    market_probs = {"home": 0.20, "draw": 0.30, "away": 0.50}
    hits = detect_model_gap(session, _match(), market_probs, xg_pred=pred)
    assert hits
    for h in hits:
        assert h.payload["league"] == "England — Premier League"
        assert h.payload["model_trust"] == "trusted"


# --- precision-gate демотирует unreliable model_gap -------------------------

def test_classify_drops_untrusted_model_gap():
    hits = [
        AnomalyHit("sharp_move", 1.0, "x", {"outcome": "home"}),
        AnomalyHit("model_gap", 1.0, "x",
                   {"outcome": "home", "market": 2.5, "fair": 2.2,
                    "model_trust": "unreliable"}),
    ]
    label, meta = classify_signal(hits)
    assert meta["dropped_untrusted_model_gap"] == 1
    # остаётся только sharp_move → 1 детектор < signal_min_detectors → weak
    assert meta["n_detectors"] == 1
    assert label == "weak"


def test_classify_keeps_trusted_model_gap():
    hits = [
        AnomalyHit("sharp_move", 1.0, "x", {"outcome": "home"}),
        AnomalyHit("model_gap", 1.0, "x",
                   {"outcome": "home", "market": 2.5, "fair": 2.2,
                    "model_trust": "trusted"}),
    ]
    _, meta = classify_signal(hits)
    assert meta["dropped_untrusted_model_gap"] == 0
    assert meta["n_detectors"] == 2
