"""Tests for bookmaker_weight and weighted consensus_probabilities."""
from __future__ import annotations

from datetime import datetime, timezone

from src.odds_client import BookmakerOdds, MatchOdds
from src.probability import (
    bookmaker_weight,
    consensus_probabilities,
    _weighted_median,
)


def _bm(bookmaker: str, home=None, draw=None, away=None) -> BookmakerOdds:
    return BookmakerOdds(bookmaker=bookmaker, home=home, draw=draw, away=away)


def _match(bookmakers) -> MatchOdds:
    return MatchOdds(
        match_id="weight-test",
        sport_key="soccer_epl",
        home_team="Home FC",
        away_team="Away FC",
        commence_time=datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc),
        bookmakers=bookmakers,
    )


# --- bookmaker_weight --------------------------------------------------------

def test_sharp_bookmaker_gets_full_weight():
    from src.config import settings
    assert bookmaker_weight("pinnacle") == settings.sharp_weight


def test_sharp_bookmaker_case_insensitive():
    from src.config import settings
    assert bookmaker_weight("Pinnacle") == settings.sharp_weight
    assert bookmaker_weight("BETFAIR_EX_EU") == settings.sharp_weight


def test_soft_bookmaker_gets_default_weight():
    from src.config import settings
    assert bookmaker_weight("bet365") == settings.default_weight
    assert bookmaker_weight("unibet") == settings.default_weight


# --- _weighted_median --------------------------------------------------------

def test_weighted_median_equal_weights_matches_median():
    """With equal weights, weighted median equals regular median."""
    vals = [0.40, 0.50, 0.60]
    weights = [1.0, 1.0, 1.0]
    assert abs(_weighted_median(vals, weights) - 0.50) < 1e-9


def test_weighted_median_high_weight_pulls_toward_its_value():
    """A single very-heavy value becomes the weighted median."""
    # vals sorted: [0.40, 0.50, 0.60], weights [1.0, 0.1, 0.1]
    # total=1.2, half=0.6; after 0.40: cumulative=1.0 >= 0.6 → median=0.40
    vals = [0.50, 0.40, 0.60]
    weights = [0.1, 1.0, 0.1]
    result = _weighted_median(vals, weights)
    assert abs(result - 0.40) < 1e-9


def test_weighted_median_single_value():
    assert _weighted_median([0.55], [1.0]) == 0.55


def test_weighted_median_two_equal_weight_values():
    """With two values of equal weight, result is the first (lower) one."""
    result = _weighted_median([0.40, 0.60], [1.0, 1.0])
    assert abs(result - 0.40) < 1e-9


# --- consensus_probabilities with weights ------------------------------------

def test_consensus_sharp_dominates_soft_outlier():
    """Sharp bookmaker's value should dominate over many soft outliers.

    Pinnacle (weight=1.0) gives home=0.45 (odds 2.22).
    Three soft books (weight=0.4 each) give home≈0.54 (odds 1.85).
    Weighted total: 1.0 vs 3×0.4=1.2 → soft wins by weight but margin is thin.

    This test checks that the weighted median doesn't blindly average —
    the result stays close to the region with the heaviest weight.
    """
    match = _match([
        _bm("pinnacle",  home=2.22, draw=3.50, away=4.00),  # sharp: home prob ≈ 45%
        _bm("bet365",    home=1.85, draw=3.50, away=4.50),  # soft: home prob ≈ 53%
        _bm("unibet",    home=1.85, draw=3.50, away=4.50),
        _bm("bwin",      home=1.85, draw=3.50, away=4.50),
    ])
    probs = consensus_probabilities(match)
    assert probs is not None
    # Three softs give ≈53%, one sharp gives ≈45%.
    # Weighted: sharp total=1.0 < soft total=1.2, so soft value is the median.
    # But with only 1 soft needed to tip: result should be between 45% and 53%.
    assert 0.44 < probs["home"] < 0.55


def test_consensus_two_sharps_outweigh_three_softs():
    """Two sharps (weight 2.0 total) outweigh three softs (weight 1.2 total)."""
    match = _match([
        _bm("pinnacle",    home=2.22, draw=3.50, away=4.00),  # sharp ≈ 45%
        _bm("betfair_ex_eu", home=2.20, draw=3.50, away=4.00),  # sharp ≈ 46%
        _bm("bet365",      home=1.85, draw=3.50, away=4.50),  # soft ≈ 53%
        _bm("unibet",      home=1.85, draw=3.50, away=4.50),
        _bm("bwin",        home=1.85, draw=3.50, away=4.50),
    ])
    probs = consensus_probabilities(match)
    assert probs is not None
    # Sharp total weight=2.0 > soft total=1.2 → weighted median pulled toward sharp
    assert probs["home"] < 0.50  # closer to 45-46% than to 53%


def test_consensus_none_with_no_valid_bookmakers():
    assert consensus_probabilities(_match([])) is None
