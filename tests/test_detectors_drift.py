"""Tests for detect_drift — sliding window, probability-based version.

Uses an in-memory SQLite database; no network calls.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base, OddsSnapshot
from src.detectors import detect_drift
from src.odds_client import BookmakerOdds, MatchOdds


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    with Session() as s:
        yield s


def _bm(bookmaker: str, home=None, draw=None, away=None) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match(bookmakers, match_id="match-1", hours_until=5.0) -> MatchOdds:
    commence = datetime.now(timezone.utc) + timedelta(hours=hours_until)
    return MatchOdds(
        match_id=match_id,
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=commence,
        bookmakers=bookmakers,
    )


def _snap(match_id, age_minutes, home=2.00, draw=3.50, away=4.00):
    """Create an OddsSnapshot `age_minutes` old with consistent bookmakers data."""
    bms = [
        {"bookmaker": "bk1", "home": home,        "draw": draw,        "away": away},
        {"bookmaker": "bk2", "home": home + 0.02, "draw": draw - 0.05, "away": away - 0.05},
        {"bookmaker": "bk3", "home": home - 0.02, "draw": draw + 0.05, "away": away + 0.05},
    ]
    return OddsSnapshot(
        match_id=match_id,
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5),
        median_home=home,
        median_draw=draw,
        median_away=away,
        bookmakers=bms,
        captured_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=age_minutes),
    )


# --- No snapshot at all ------------------------------------------------------

def test_drift_no_snapshot_returns_empty(session):
    """First poll — no snapshot in DB → nothing to compare against."""
    match = _match([_bm("bk1", home=2.00, draw=3.50, away=4.00)])
    assert detect_drift(session, match) == []


# --- Snapshot within window → should detect large drift ----------------------

def test_drift_detects_large_movement_within_window(session):
    """90-min-old snapshot (inside 120-min window) with large home drift → fires."""
    # Opening: home prob ≈ 46% (home=2.20 opening odds)
    session.add(_snap("match-1", age_minutes=90, home=2.20, draw=3.50, away=4.00))
    session.commit()

    # Current: home prob ≈ 53% (home=1.65, shift ≈ 7pp > 5pp threshold)
    match = _match([
        _bm("bk1", home=1.65, draw=3.50, away=4.00),
        _bm("bk2", home=1.67, draw=3.45, away=3.95),
        _bm("bk3", home=1.63, draw=3.55, away=4.05),
    ], match_id="match-1")

    hits = detect_drift(session, match)
    home_hits = [h for h in hits if h.detector == "drift" and "home" in h.description]
    assert home_hits, "Expected drift hit on home"
    assert home_hits[0].severity > 5.0
    # Payload should contain window info
    p = home_hits[0].payload
    assert "window_minutes" in p
    assert "opening_prob" in p
    assert "current_prob" in p
    assert p["drift_pp"] > 0  # prob went up (odds fell) → drift_pp = current - opening > 0


def test_drift_silent_on_small_movement(session):
    """90-min-old snapshot with tiny drift → no hit."""
    session.add(_snap("match-2", age_minutes=90, home=2.00, draw=3.50, away=4.00))
    session.commit()

    # Current odds barely changed
    match = _match([
        _bm("bk1", home=1.98, draw=3.52, away=4.02),
        _bm("bk2", home=2.00, draw=3.50, away=4.00),
        _bm("bk3", home=1.99, draw=3.51, away=4.01),
    ], match_id="match-2")

    assert detect_drift(session, match) == []


# --- Snapshot outside window → should be ignored ----------------------------

def test_drift_ignores_snapshot_outside_window(session):
    """Snapshot from 5 hours ago falls outside the 120-min window → return []."""
    session.add(_snap("match-3", age_minutes=300, home=2.00, draw=3.50, away=4.00))
    session.commit()

    # Current odds very different (would trigger if in window)
    match = _match([
        _bm("bk1", home=1.65, draw=3.50, away=4.00),
        _bm("bk2", home=1.67, draw=3.45, away=3.95),
        _bm("bk3", home=1.63, draw=3.55, away=4.05),
    ], match_id="match-3")

    hits = detect_drift(session, match)
    assert hits == [], (
        f"Expected no hits: 5-hour-old snapshot is outside the 120-min window, "
        f"got {[h.description for h in hits]}"
    )


# --- Window boundary: uses oldest snapshot within window ---------------------

def test_drift_uses_oldest_snapshot_in_window(session):
    """With two snapshots in window (30min and 90min old), uses the 90-min one."""
    # 90-min old: opening home odds 2.20 → opening_prob ≈ 0.459
    session.add(_snap("match-4", age_minutes=90, home=2.20, draw=3.50, away=4.00))
    # 30-min old: opening home odds 1.80 → opening_prob ≈ 0.519 (closer to current)
    session.add(_snap("match-4", age_minutes=30,  home=1.80, draw=3.50, away=4.00))
    session.commit()

    # Current: home≈1.65 → prob ≈ 0.531
    # Drift from 2.20 snap: ≈7.2pp (triggers)
    # Drift from 1.80 snap: ≈1.2pp (would NOT trigger if wrongly using 30-min snap)
    match = _match([
        _bm("bk1", home=1.65, draw=3.50, away=4.00),
        _bm("bk2", home=1.67, draw=3.45, away=3.95),
        _bm("bk3", home=1.63, draw=3.55, away=4.05),
    ], match_id="match-4")

    hits = detect_drift(session, match)
    home_hits = [h for h in hits if "home" in h.description]
    assert home_hits, "Expected drift hit: oldest (2.20) snapshot should be used"
    # opening_prob from 2.20 snap ≈ 0.459, well below 0.519 of 1.80 snap
    assert home_hits[0].payload["opening_prob"] < 0.48
