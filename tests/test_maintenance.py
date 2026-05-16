"""Тест выборочного сброса: чистим detector-таблицы, бережём остальное."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import src.maintenance as mnt
from src.db import (Anomaly, AnomalyCLV, AnomalyOutcome, Base, MatchResult,
                    OddsSnapshot, ProbCalibration, ResultNotification,
                    SstatsModelOutcome, TeamRating)


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(mnt, "SessionLocal", Session)
    return Session


def test_purge_clears_detector_keeps_model(db):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with db() as s:
        s.add(Anomaly(match_id="m", detected_at=now, home_team="A",
                       away_team="B", commence_time=now, detector="spread",
                       severity=1.0, details="x", payload={}))
        s.add(OddsSnapshot(match_id="m", sport_key="s", home_team="A",
                           away_team="B", commence_time=now, bookmakers=[]))
        s.add(AnomalyCLV(anomaly_id=1, detector="spread"))
        s.add(AnomalyOutcome(result_key="rk", detector="spread", confirmed=1))
        s.add(ProbCalibration(match_id="m", actual="home", brier=0.5,
                              log_loss=1.0, p_home=0.5, p_draw=0.3,
                              p_away=0.2))
        s.add(ResultNotification(result_key="rk", result_found=True))
        # сохраняемые
        s.add(SstatsModelOutcome(game_id=1, league="L", home_team="A",
                                 away_team="B", p_home=0.4, p_draw=0.3,
                                 p_away=0.3, actual="home", brier=0.5,
                                 log_loss=1.0))
        s.add(MatchResult(result_key="rk2", home_team="A", away_team="B",
                          home_score=1, away_score=0))
        s.add(TeamRating(team="A", rating=1500.0))
        s.commit()

    deleted = mnt.purge_detector_data()
    assert deleted["anomalies"] == 1 and deleted["odds_snapshots"] == 1
    assert deleted["anomaly_clv"] == 1 and deleted["prob_calibration"] == 1

    with db() as s:
        for model in (Anomaly, OddsSnapshot, AnomalyCLV, AnomalyOutcome,
                      ProbCalibration, ResultNotification):
            assert s.scalar(select(func.count()).select_from(model)) == 0
        # сохранено
        assert s.scalar(select(func.count()).select_from(SstatsModelOutcome)) == 1
        assert s.scalar(select(func.count()).select_from(MatchResult)) == 1
        assert s.scalar(select(func.count()).select_from(TeamRating)) == 1


def test_purge_idempotent(db):
    mnt.purge_detector_data()
    assert sum(mnt.purge_detector_data().values()) == 0
