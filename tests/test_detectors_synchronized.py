"""Tests for detect_synchronized — sharp-bookmaker filter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base, OddsSnapshot
from src.detectors import detect_synchronized
from src.odds_client import BookmakerOdds, MatchOdds


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    with Session() as s:
        yield s


def _bm(bookmaker: str, home: float, draw: float = 3.50, away: float = 4.00) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match(bookmakers, match_id="sync-test") -> MatchOdds:
    return MatchOdds(
        match_id=match_id,
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=12),
        bookmakers=bookmakers,
    )


def _prev_snap(session, match_id: str, bm_prices: dict[str, float]) -> None:
    """Insert a previous OddsSnapshot with given home prices per bookmaker."""
    bms = [
        {"bookmaker": bm, "home": price, "draw": 3.50, "away": 4.00}
        for bm, price in bm_prices.items()
    ]
    session.add(OddsSnapshot(
        match_id=match_id,
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=12),
        median_home=2.00, median_draw=3.50, median_away=4.00,
        bookmakers=bms,
        captured_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=15),
    ))
    session.commit()


# --- No previous snapshot ---------------------------------------------------

def test_no_previous_snapshot_returns_empty(session):
    match = _match([_bm("pinnacle", 1.86)])
    assert detect_synchronized(session, match) == []


# --- 4 soft bookmakers only → should NOT fire --------------------------------

def test_four_followers_only_silent(session):
    """4 soft bookmakers all drop home odds 7% → no sharp → no hit."""
    soft_books = ["bet365", "unibet", "bwin", "coral"]
    _prev_snap(session, "sync-test", {bm: 2.00 for bm in soft_books})

    # All move 7% down (exceeds 5% sync_move_threshold)
    match = _match([_bm(bm, home=1.86) for bm in soft_books])
    hits = detect_synchronized(session, match)

    assert not hits, (
        f"Expected no hit: 4 soft-only movers should be filtered, got {[h.description for h in hits]}"
    )


# --- 3 soft + 1 sharp → should fire -----------------------------------------

def test_three_followers_plus_one_sharp_fires(session):
    """3 soft + pinnacle all drop home odds 7% → pinnacle is sharp → fires."""
    books = {"bet365": 2.00, "unibet": 2.00, "bwin": 2.00, "pinnacle": 2.00}
    _prev_snap(session, "sync-2", books)

    match = _match(
        [_bm(bm, home=1.86) for bm in books],
        match_id="sync-2",
    )
    hits = detect_synchronized(session, match)

    sync_hits = [h for h in hits if h.detector == "synchronized"]
    assert sync_hits, "Expected synchronized hit when pinnacle is among movers"

    hit = sync_hits[0]
    assert "pinnacle" in hit.description
    assert "pinnacle" in hit.payload["sharp_movers"]


# --- 1 sharp alone (below min_movers) → should NOT fire ----------------------

def test_only_sharp_below_min_movers_silent(session):
    """Only pinnacle moves (1 bookmaker < min_movers=3) → no hit."""
    books = {"pinnacle": 2.00, "bet365": 2.00}
    _prev_snap(session, "sync-3", books)

    # Only pinnacle moves; bet365 stays flat
    match = _match(
        [_bm("pinnacle", home=1.86), _bm("bet365", home=2.00)],
        match_id="sync-3",
    )
    hits = detect_synchronized(session, match)
    assert not hits, "Expected no hit: only 1 mover below min_movers=3"


# --- Payload correctness -----------------------------------------------------

def test_payload_has_sharp_movers_field(session):
    """sharp_movers list in payload names which sharp books triggered."""
    books = {"pinnacle": 2.00, "matchbook": 2.00, "bet365": 2.00}
    _prev_snap(session, "sync-4", books)

    match = _match([_bm(bm, home=1.86) for bm in books], match_id="sync-4")
    hits = detect_synchronized(session, match)
    sync_hits = [h for h in hits if h.detector == "synchronized"]
    assert sync_hits

    sharps = sync_hits[0].payload["sharp_movers"]
    assert "pinnacle" in sharps or "matchbook" in sharps
