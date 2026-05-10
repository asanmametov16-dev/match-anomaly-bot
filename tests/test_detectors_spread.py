"""Tests for detect_spread — probability-based version.

All tests use synthetic MatchOdds with no network calls.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.odds_client import BookmakerOdds, MatchOdds
from src.detectors import detect_spread


def _bm(bookmaker: str, home=None, draw=None, away=None) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match(bookmakers: list[BookmakerOdds]) -> MatchOdds:
    return MatchOdds(
        match_id="test-spread",
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc),
        bookmakers=bookmakers,
    )


# --- Positive: large spread should trigger -----------------------------------

def test_spread_triggers_on_large_home_spread():
    """Bookmakers disagree significantly on home win probability → fires."""
    # bk1: home=2.20 → clean_home ≈ 43%
    # bk2: home=1.70 → clean_home ≈ 54%
    # bk3: home=1.75 → clean_home ≈ 52%
    # Spread ≈ 11pp >> 4pp threshold
    match = _match([
        _bm("bk1", home=2.20, draw=3.50, away=3.20),
        _bm("bk2", home=1.70, draw=3.50, away=4.50),
        _bm("bk3", home=1.75, draw=3.50, away=4.20),
    ])
    hits = detect_spread(match)
    home_hits = [h for h in hits if h.detector == "spread" and "home" in h.description]
    assert home_hits, "Expected a spread hit on home outcome"
    assert home_hits[0].severity > 4.0
    # Description should include both pp and raw odds
    desc = home_hits[0].description
    assert "пп" in desc
    assert "%" in desc


# --- Negative: tight spread should be silent ---------------------------------

def test_spread_silent_on_tight_spread():
    """Bookmakers with nearly identical home odds → no spread detected."""
    # All three give home probability within ~1pp of each other
    match = _match([
        _bm("bk1", home=1.95, draw=3.60, away=4.00),
        _bm("bk2", home=1.97, draw=3.55, away=4.00),
        _bm("bk3", home=1.96, draw=3.58, away=4.05),
    ])
    hits = detect_spread(match)
    assert not any(h.detector == "spread" for h in hits), (
        f"Unexpected spread hit: {[h.description for h in hits]}"
    )


# --- At least 3 bookmakers required -----------------------------------------

def test_spread_requires_three_bookmakers():
    """With only 2 bookmakers, even a large spread should be ignored."""
    match = _match([
        _bm("bk1", home=2.20, draw=3.50, away=3.20),
        _bm("bk2", home=1.70, draw=3.50, away=4.50),
    ])
    hits = detect_spread(match)
    assert not any(h.detector == "spread" for h in hits)


# --- Consistency across probability levels -----------------------------------

def test_spread_consistent_at_low_probability():
    """4+pp spread on away outsider (~10% vs ~14%) also triggers.

    The old relative-% detector would treat this 40%-relative spread
    very differently from a 8%-relative spread at 50% probability.
    The new pp-based detector fires equally for the same absolute gap.
    """
    # bk1: away=10.0 → clean_away ≈ 9.8%
    # bk2: away=6.50 → clean_away ≈ 14.4%
    # bk3: away=8.00 → clean_away ≈ 12.0%
    # Spread ≈ 4.6pp → should trigger
    match = _match([
        _bm("bk1", home=1.50, draw=4.00, away=10.0),
        _bm("bk2", home=1.50, draw=4.00, away=6.50),
        _bm("bk3", home=1.50, draw=4.00, away=8.00),
    ])
    hits = detect_spread(match)
    away_hits = [h for h in hits if h.detector == "spread" and "away" in h.description]
    assert away_hits, "Expected spread hit on away at low probability level"


def test_spread_consistent_at_high_probability():
    """Same ~4pp spread at home probability ~50% also triggers."""
    # bk1: home=2.00, draw=3.60, away=4.50 → clean_home ≈ 50.0%
    # bk2: home=1.70, draw=3.60, away=5.00 → clean_home ≈ 55.2%
    # bk3: home=1.85, draw=3.60, away=4.80 → clean_home ≈ 52.7%
    # Spread ≈ 5.2pp → should trigger
    match = _match([
        _bm("bk1", home=2.00, draw=3.60, away=4.50),
        _bm("bk2", home=1.70, draw=3.60, away=5.00),
        _bm("bk3", home=1.85, draw=3.60, away=4.80),
    ])
    hits = detect_spread(match)
    home_hits = [h for h in hits if h.detector == "spread" and "home" in h.description]
    assert home_hits, "Expected spread hit on home at high probability level"


# --- Payload correctness -----------------------------------------------------

def test_spread_payload_contains_expected_keys():
    """Payload must have lo/hi bookmaker names, probs, and odds."""
    match = _match([
        _bm("bk1", home=2.20, draw=3.50, away=3.20),
        _bm("bk2", home=1.70, draw=3.50, away=4.50),
        _bm("bk3", home=1.75, draw=3.50, away=4.20),
    ])
    hits = detect_spread(match)
    home_hit = next(h for h in hits if "home" in h.description)
    p = home_hit.payload
    for key in ("outcome", "lo_bm", "lo_prob", "lo_odds", "hi_bm", "hi_prob", "hi_odds", "spread_pp"):
        assert key in p, f"Missing key '{key}' in payload"
