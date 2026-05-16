"""Tests for CLV (closing line value) computation.

Uses in-memory SQLite; no network. Covers:
- bet-side extraction per detector type
- end-to-end CLV computation with synthetic snapshots
- idempotent re-runs (compute_pending_clv processes each anomaly once)
- non-directional detectors produce a sentinel row instead of being skipped
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import src.clv as clv_module
from src.clv import compute_clv_for_anomaly, compute_pending_clv, extract_bet_side
from src.db import Anomaly, AnomalyCLV, Base, OddsSnapshot


# --- Unit: bet-side extraction ----------------------------------------------

class TestExtractBetSide:
    def test_drift_positive_returns_outcome(self):
        assert extract_bet_side("drift", {"outcome": "home", "drift_pp": 5.0}) == "home"

    def test_drift_negative_returns_none(self):
        assert extract_bet_side("drift", {"outcome": "home", "drift_pp": -5.0}) is None

    def test_drift_zero_returns_none(self):
        assert extract_bet_side("drift", {"outcome": "home", "drift_pp": 0}) is None

    def test_synchronized_down_returns_outcome(self):
        assert extract_bet_side("synchronized",
                                {"outcome": "away", "direction": "↓"}) == "away"

    def test_synchronized_up_returns_none(self):
        assert extract_bet_side("synchronized",
                                {"outcome": "away", "direction": "↑"}) is None

    def test_model_gap_market_higher_than_fair(self):
        # market 2.50 > fair 2.20 → market underprices outcome → bet outcome
        assert extract_bet_side(
            "model_gap", {"outcome": "home", "market": 2.50, "fair": 2.20}
        ) == "home"

    def test_model_gap_market_lower_than_fair(self):
        assert extract_bet_side(
            "model_gap", {"outcome": "home", "market": 2.00, "fair": 2.20}
        ) is None

    def test_sharp_move_returns_outcome(self):
        assert extract_bet_side("sharp_move", {"outcome": "draw"}) == "draw"

    def test_spread_returns_none(self):
        assert extract_bet_side(
            "spread", {"outcome": "home", "spread_pp": 5.0}
        ) is None

    def test_exotic_spread_returns_none(self):
        assert extract_bet_side("exotic_spread", {"outcome": "home"}) is None

    def test_missing_outcome_returns_none(self):
        assert extract_bet_side("drift", {"drift_pp": 5.0}) is None

    def test_invalid_outcome_returns_none(self):
        assert extract_bet_side("drift", {"outcome": "xxx", "drift_pp": 5.0}) is None

    def test_none_payload_returns_none(self):
        assert extract_bet_side("drift", None) is None


# --- Integration: full CLV computation --------------------------------------

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    with Session() as s:
        yield s


def _bms(home: float, draw: float, away: float):
    """Synthetic bookmakers — include one sharp + soft for weighted consensus."""
    return [
        {"bookmaker": "pinnacle",      "home": home,        "draw": draw,        "away": away},
        {"bookmaker": "betfair_ex_eu", "home": home + 0.01, "draw": draw,        "away": away - 0.02},
        {"bookmaker": "bk3",           "home": home + 0.03, "draw": draw - 0.05, "away": away + 0.05},
        {"bookmaker": "bk4",           "home": home - 0.02, "draw": draw + 0.05, "away": away - 0.03},
    ]


def _snap(match_id: str, captured_at: datetime, commence: datetime,
          home: float, draw: float, away: float) -> OddsSnapshot:
    return OddsSnapshot(
        match_id=match_id, sport_key="soccer_epl",
        home_team="Home FC", away_team="Away FC",
        commence_time=commence,
        median_home=home, median_draw=draw, median_away=away,
        bookmakers=_bms(home, draw, away),
        captured_at=captured_at,
    )


def _now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_compute_clv_drift_positive_alpha(session):
    """Alert says home prob ↑; market keeps moving home prob ↑ → positive CLV."""
    now = _now_naive()
    commence = now - timedelta(hours=1)
    alert_t = now - timedelta(hours=3)

    # Alert snapshot: home=2.00 → margin-free prob ≈ 51%
    session.add(_snap("m1", captured_at=alert_t, commence=commence,
                      home=2.00, draw=3.50, away=4.00))
    # Closing snapshot: home dropped to 1.75 → prob ≈ 56%
    session.add(_snap("m1", captured_at=commence - timedelta(minutes=5),
                      commence=commence, home=1.75, draw=3.60, away=4.50))

    a = Anomaly(
        match_id="m1", detected_at=alert_t,
        home_team="Home FC", away_team="Away FC",
        commence_time=commence, detector="drift",
        severity=5.0, details="test", payload={"outcome": "home", "drift_pp": 5.0},
    )
    session.add(a)
    session.flush()

    row = compute_clv_for_anomaly(session, a)
    assert row.side == "home"
    assert row.prob_at_alert is not None
    assert row.prob_at_close is not None
    assert row.clv_pp > 0, f"expected positive CLV, got {row.clv_pp}"


def test_compute_clv_sharp_move_negative_when_market_reverts(session):
    """Sharp said home up; close moved BACK toward original → negative CLV."""
    now = _now_naive()
    commence = now - timedelta(hours=1)
    alert_t = now - timedelta(hours=3)

    session.add(_snap("m2", captured_at=alert_t, commence=commence,
                      home=1.80, draw=3.50, away=4.50))
    # Closing reverted: home odds back up → home prob DOWN
    session.add(_snap("m2", captured_at=commence - timedelta(minutes=5),
                      commence=commence, home=2.10, draw=3.40, away=3.80))

    a = Anomaly(
        match_id="m2", detected_at=alert_t,
        home_team="Home FC", away_team="Away FC",
        commence_time=commence, detector="sharp_move",
        severity=0.05, details="test", payload={"outcome": "home"},
    )
    session.add(a)
    session.flush()

    row = compute_clv_for_anomaly(session, a)
    assert row.side == "home"
    assert row.clv_pp < 0


def test_compute_clv_non_directional_writes_null_row(session):
    """spread detector → side=None, clv_pp=None (sentinel)."""
    now = _now_naive()
    a = Anomaly(
        match_id="m3", detected_at=now - timedelta(hours=3),
        home_team="Home FC", away_team="Away FC",
        commence_time=now - timedelta(hours=1), detector="spread",
        severity=5.0, details="test",
        payload={"outcome": "home", "spread_pp": 5.0},
    )
    session.add(a)
    session.flush()

    row = compute_clv_for_anomaly(session, a)
    assert row.side is None
    assert row.clv_pp is None


def test_compute_clv_missing_snapshots_writes_null_clv(session):
    """Directional detector but no snapshots in DB → side set, clv_pp NULL."""
    now = _now_naive()
    a = Anomaly(
        match_id="missing", detected_at=now - timedelta(hours=3),
        home_team="Home FC", away_team="Away FC",
        commence_time=now - timedelta(hours=1), detector="drift",
        severity=5.0, details="test", payload={"outcome": "home", "drift_pp": 5.0},
    )
    session.add(a)
    session.flush()

    row = compute_clv_for_anomaly(session, a)
    assert row.side == "home"
    assert row.prob_at_alert is None
    assert row.clv_pp is None


def test_compute_pending_clv_only_past_kickoff(session, monkeypatch):
    """Only anomalies whose match has kicked off are processed."""
    now = _now_naive()

    past = Anomaly(
        match_id="past", detected_at=now - timedelta(hours=3),
        home_team="Home FC", away_team="Away FC",
        commence_time=now - timedelta(hours=1),
        detector="spread", severity=5.0, details="t",
        payload={"outcome": "home"},
    )
    future = Anomaly(
        match_id="future", detected_at=now,
        home_team="Home FC", away_team="Away FC",
        commence_time=now + timedelta(hours=2),
        detector="spread", severity=5.0, details="t",
        payload={"outcome": "home"},
    )
    session.add_all([past, future])
    session.commit()

    # compute_pending_clv opens its own SessionLocal — point it at our in-memory engine
    monkeypatch.setattr(clv_module, "SessionLocal",
                        sessionmaker(bind=session.get_bind(), autoflush=False))

    written = compute_pending_clv()
    assert written == 1

    rows = session.execute(select(AnomalyCLV)).scalars().all()
    assert len(rows) == 1
    assert rows[0].anomaly_id == past.id


def test_compute_pending_clv_idempotent(session, monkeypatch):
    """Running twice in a row processes each anomaly exactly once."""
    now = _now_naive()
    a = Anomaly(
        match_id="x", detected_at=now - timedelta(hours=3),
        home_team="Home FC", away_team="Away FC",
        commence_time=now - timedelta(hours=1),
        detector="spread", severity=5.0, details="t",
        payload={"outcome": "home"},
    )
    session.add(a)
    session.commit()

    monkeypatch.setattr(clv_module, "SessionLocal",
                        sessionmaker(bind=session.get_bind(), autoflush=False))

    assert compute_pending_clv() == 1
    assert compute_pending_clv() == 0
    rows = session.execute(select(AnomalyCLV)).scalars().all()
    assert len(rows) == 1
