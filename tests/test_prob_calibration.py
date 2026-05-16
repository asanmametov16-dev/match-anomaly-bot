"""Тесты калибровки вероятностей против исходов (Brier/log-loss).

In-memory SQLite, без сети. Покрывает математику, end-to-end расчёт,
идемпотентность, sentinel при отсутствии результата и агрегат.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import src.prob_calibration as pc
from src.db import Base, MatchResult, OddsSnapshot, ProbCalibration
from src.prob_calibration import (brier_score, calibration_summary,
                                  compute_pending_calibration, log_loss,
                                  reliability_bins)
from src.result_checker import _result_key


# --- математика --------------------------------------------------------------

def test_brier_perfect_is_zero():
    assert brier_score({"home": 1.0, "draw": 0.0, "away": 0.0}, "home") == pytest.approx(0.0)


def test_brier_uniform_is_two_thirds():
    u = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    assert brier_score(u, "home") == pytest.approx(2.0 / 3.0)


def test_brier_worst_case():
    assert brier_score({"home": 0.0, "draw": 0.0, "away": 1.0}, "home") == pytest.approx(2.0)


def test_log_loss_perfect_and_clipped():
    assert log_loss({"home": 1.0, "draw": 0.0, "away": 0.0}, "home") == pytest.approx(0.0)
    # p=0 на факт → клип 1e-12, конечное большое число
    assert log_loss({"home": 0.0, "draw": 0.0, "away": 1.0}, "home") == pytest.approx(-math.log(1e-12))


def test_reliability_bins_split_and_freq():
    pts = [(0.05, 0), (0.07, 0), (0.95, 1), (0.92, 1)]
    bins = reliability_bins(pts, n_bins=10)
    assert len(bins) == 2
    lo_bin = next(b for b in bins if b["lo"] == 0.0)
    hi_bin = next(b for b in bins if b["lo"] == 0.9)
    assert lo_bin["n"] == 2 and lo_bin["emp_freq"] == pytest.approx(0.0)
    assert hi_bin["n"] == 2 and hi_bin["emp_freq"] == pytest.approx(1.0)


# --- end-to-end --------------------------------------------------------------

@pytest.fixture
def session(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(pc, "SessionLocal", sessionmaker(bind=engine, autoflush=False))
    with Session() as s:
        yield s


def _now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _bms(home: float, draw: float, away: float):
    return [
        {"bookmaker": "pinnacle",      "home": home,        "draw": draw,        "away": away},
        {"bookmaker": "betfair_ex_eu", "home": home + 0.01, "draw": draw,        "away": away - 0.02},
        {"bookmaker": "bk3",           "home": home + 0.03, "draw": draw - 0.05, "away": away + 0.05},
    ]


def _add_snapshot(s, match_id, home, away, commence, captured_at,
                  odds=(1.80, 3.60, 4.50)):
    s.add(OddsSnapshot(
        match_id=match_id, sport_key="soccer_epl",
        home_team=home, away_team=away, commence_time=commence,
        median_home=odds[0], median_draw=odds[1], median_away=odds[2],
        bookmakers=_bms(*odds), captured_at=captured_at,
    ))


def _add_result(s, home, away, commence, hs, as_):
    s.add(MatchResult(
        result_key=_result_key(home, away, commence),
        home_team=home, away_team=away,
        home_score=hs, away_score=as_, competition="Test",
    ))


def test_scores_finished_match(session):
    now = _now_naive()
    commence = now - timedelta(hours=5)
    _add_snapshot(session, "m1", "Arsenal", "Chelsea", commence,
                  captured_at=commence - timedelta(minutes=5))
    _add_result(session, "Arsenal", "Chelsea", commence, hs=2, as_=0)  # home win
    session.commit()

    assert compute_pending_calibration() == 1

    row = session.execute(select(ProbCalibration)).scalar_one()
    assert row.match_id == "m1"
    assert row.actual == "home"
    assert row.brier is not None and 0.0 <= row.brier <= 2.0
    # brier должен соответствовать сохранённым вероятностям
    probs = {"home": row.p_home, "draw": row.p_draw, "away": row.p_away}
    assert row.brier == pytest.approx(brier_score(probs, "home"))
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_idempotent(session):
    now = _now_naive()
    commence = now - timedelta(hours=5)
    _add_snapshot(session, "m1", "A", "B", commence,
                  captured_at=commence - timedelta(minutes=5))
    _add_result(session, "A", "B", commence, hs=1, as_=1)  # draw
    session.commit()

    assert compute_pending_calibration() == 1
    assert compute_pending_calibration() == 0
    assert len(session.execute(select(ProbCalibration)).scalars().all()) == 1


def test_sentinel_when_no_result_and_old(session):
    now = _now_naive()
    commence = now - timedelta(hours=100)  # >96ч, результата нет
    _add_snapshot(session, "old", "X", "Y", commence,
                  captured_at=commence - timedelta(minutes=5))
    session.commit()

    assert compute_pending_calibration() == 1
    row = session.execute(select(ProbCalibration)).scalar_one()
    assert row.actual is None and row.brier is None
    # повторный прогон не пересчитывает sentinel
    assert compute_pending_calibration() == 0


def test_waits_when_no_result_but_recent(session):
    now = _now_naive()
    commence = now - timedelta(hours=5)  # <96ч, результата ещё нет
    _add_snapshot(session, "wait", "X", "Y", commence,
                  captured_at=commence - timedelta(minutes=5))
    session.commit()

    assert compute_pending_calibration() == 0
    assert session.execute(select(ProbCalibration)).first() is None


def test_calibration_summary(session):
    now = _now_naive()
    for i, (hs, as_) in enumerate([(2, 0), (0, 1), (1, 1)]):
        c = now - timedelta(hours=5 + i)
        _add_snapshot(session, f"m{i}", f"H{i}", f"A{i}", c,
                      captured_at=c - timedelta(minutes=5))
        _add_result(session, f"H{i}", f"A{i}", c, hs, as_)
    session.commit()

    compute_pending_calibration()
    s = calibration_summary()
    assert s["n"] == 3
    assert 0.0 <= s["mean_brier"] <= 2.0
    assert s["uniform_brier"] == pytest.approx(2 / 3)
    assert s["bins"]  # непустая кривая
