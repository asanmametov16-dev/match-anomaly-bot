"""Tests for src/probability.py — no network calls, pure logic."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.odds_client import BookmakerOdds, MatchOdds
from src.probability import (
    consensus_probabilities,
    implied_probability,
    probabilities_from_match,
    probabilities_to_odds,
    remove_overround,
)


def _bm(bookmaker: str = "test", home=None, draw=None, away=None) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match(bookmakers: list[BookmakerOdds]) -> MatchOdds:
    return MatchOdds(
        match_id="test-match",
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc),
        bookmakers=bookmakers,
    )


# --- implied_probability -------------------------------------------------------

def test_implied_probability_basic():
    assert abs(implied_probability(2.0) - 0.5) < 1e-9


def test_implied_probability_raises_on_le_one():
    with pytest.raises(ValueError):
        implied_probability(1.0)
    with pytest.raises(ValueError):
        implied_probability(0.5)


# --- remove_overround ----------------------------------------------------------

def test_remove_overround_sums_to_one():
    # 1.90/3.50/4.20 → implied total ≈ 1.05 (5% margin)
    raw = {
        "home": 1 / 1.90,
        "draw": 1 / 3.50,
        "away": 1 / 4.20,
    }
    total_raw = sum(raw.values())
    assert 1.04 < total_raw < 1.06, f"expected ~1.05 overround, got {total_raw:.4f}"

    clean = remove_overround(raw)
    assert abs(sum(clean.values()) - 1.0) < 1e-9


def test_remove_overround_preserves_order():
    raw = {"home": 0.60, "draw": 0.25, "away": 0.30}
    clean = remove_overround(raw)
    assert clean["home"] > clean["away"] > clean["draw"]


# --- probabilities_from_match --------------------------------------------------

def test_probabilities_from_match_basic():
    bm = _bm(home=1.90, draw=3.50, away=4.20)
    probs = probabilities_from_match(bm)
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert set(probs) == {"home", "draw", "away"}


def test_probabilities_from_match_none_when_no_prices():
    assert probabilities_from_match(_bm()) is None


def test_probabilities_from_match_none_when_only_one_price():
    # Need at least 2 outcomes to compute overround
    assert probabilities_from_match(_bm(home=1.90)) is None


def test_probabilities_from_match_works_without_draw():
    # Some markets (e.g. moneyline) have no draw
    bm = _bm(home=1.90, away=2.10)
    probs = probabilities_from_match(bm)
    assert probs is not None
    assert set(probs) == {"home", "away"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


# --- consensus_probabilities ---------------------------------------------------

def test_consensus_probabilities_basic():
    match = _match([
        _bm("bk1", home=1.90, draw=3.50, away=4.20),
        _bm("bk2", home=1.85, draw=3.60, away=4.50),
        _bm("bk3", home=1.95, draw=3.40, away=4.00),
    ])
    probs = consensus_probabilities(match)
    assert probs is not None
    # Home is clear favourite
    assert probs["home"] > probs["draw"]
    assert probs["home"] > probs["away"]
    # All values are valid probabilities
    for v in probs.values():
        assert 0.0 < v < 1.0


def test_consensus_probabilities_uses_median():
    # bk3 is outlier with very low home odds — median should ignore it
    match = _match([
        _bm("bk1", home=2.00, draw=3.50, away=3.50),
        _bm("bk2", home=2.00, draw=3.50, away=3.50),
        _bm("bk3", home=1.10, draw=5.00, away=8.00),  # outlier
    ])
    probs = consensus_probabilities(match)
    assert probs is not None
    bk1 = probabilities_from_match(_bm("bk1", home=2.00, draw=3.50, away=3.50))
    # Median home prob should be close to bk1/bk2, not dragged up by outlier
    assert abs(probs["home"] - bk1["home"]) < 0.02


def test_consensus_probabilities_none_when_empty():
    assert consensus_probabilities(_match([])) is None


def test_consensus_probabilities_none_when_no_valid_bookmakers():
    match = _match([_bm(), _bm()])  # no prices
    assert consensus_probabilities(match) is None


# --- probabilities_to_odds -----------------------------------------------------

def test_probabilities_to_odds_roundtrip():
    original = {"home": 0.50, "draw": 0.25, "away": 0.25}
    odds = probabilities_to_odds(original)
    assert abs(odds["home"] - 2.0) < 1e-9
    assert abs(odds["draw"] - 4.0) < 1e-9
    assert abs(odds["away"] - 4.0) < 1e-9
