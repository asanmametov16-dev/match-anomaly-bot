"""Tests for time-bucket threshold scaling in detectors.

The same signal should fire in the <6h bucket (lower effective threshold)
but not in the >72h bucket (higher effective threshold).

Spread base threshold = 4.0pp:
  <6h  multiplier 0.70 → effective 2.8pp
  >72h multiplier 1.30 → effective 5.2pp
We use a synthetic spread of ~4.2pp that falls between these two values.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.odds_client import BookmakerOdds, MatchOdds
from src.detectors import detect_spread, _time_bucket_info


def _bm(bookmaker: str, home=None, draw=None, away=None) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match_in_bucket(bookmakers, hours_until: float) -> MatchOdds:
    return MatchOdds(
        match_id="bucket-test",
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=hours_until),
        bookmakers=bookmakers,
    )


# Bookmakers that give ~4.2pp home spread:
#   bk1: home=2.15 → clean_home ≈ 45.9%
#   bk2: home=1.90 → clean_home ≈ 50.1%
#   bk3: home=2.00 → clean_home ≈ 48.3%
#   spread ≈ 4.2pp  (2.8pp < 4.2pp < 5.2pp)
BORDERLINE_BMS = [
    _bm("bk1", home=2.15, draw=3.50, away=3.80),
    _bm("bk2", home=1.90, draw=3.50, away=4.20),
    _bm("bk3", home=2.00, draw=3.50, away=4.00),
]


# --- _time_bucket_info correctness -------------------------------------------

def test_bucket_under_6h():
    match = _match_in_bucket([], hours_until=3.0)
    mult, label = _time_bucket_info(match)
    assert abs(mult - 0.70) < 1e-9
    assert "<6ч" in label


def test_bucket_6_to_24h():
    match = _match_in_bucket([], hours_until=12.0)
    mult, label = _time_bucket_info(match)
    assert abs(mult - 0.85) < 1e-9
    assert "<24ч" in label


def test_bucket_24_to_72h():
    match = _match_in_bucket([], hours_until=48.0)
    mult, label = _time_bucket_info(match)
    assert abs(mult - 1.0) < 1e-9
    assert "<72ч" in label


def test_bucket_over_72h():
    match = _match_in_bucket([], hours_until=100.0)
    mult, label = _time_bucket_info(match)
    assert abs(mult - 1.3) < 1e-9
    assert ">72ч" in label


# --- Same signal fires <6h but not >72h --------------------------------------

def test_borderline_spread_fires_under_6h():
    """~4.2pp spread exceeds effective threshold 2.8pp in the <6h bucket."""
    match = _match_in_bucket(BORDERLINE_BMS, hours_until=3.0)
    hits = detect_spread(match)
    home_hits = [h for h in hits if "home" in h.description]
    assert home_hits, (
        "Expected spread hit in <6h bucket (eff. threshold 2.8pp vs ~4.2pp spread)"
    )


def test_borderline_spread_silent_over_72h():
    """Same ~4.2pp spread is below effective threshold 5.2pp in the >72h bucket."""
    match = _match_in_bucket(BORDERLINE_BMS, hours_until=100.0)
    hits = detect_spread(match)
    home_hits = [h for h in hits if "home" in h.description]
    assert not home_hits, (
        f"Expected no spread hit in >72h bucket (eff. threshold 5.2pp vs ~4.2pp spread), "
        f"got: {[h.description for h in home_hits]}"
    )


# --- Description includes bucket info ----------------------------------------

def test_description_includes_bucket_label():
    """AnomalyHit description must mention the time bucket and effective threshold."""
    match = _match_in_bucket(BORDERLINE_BMS, hours_until=3.0)
    hits = detect_spread(match)
    home_hits = [h for h in hits if "home" in h.description]
    assert home_hits
    desc = home_hits[0].description
    assert "корзина" in desc
    assert "эфф." in desc
    assert "пп" in desc
