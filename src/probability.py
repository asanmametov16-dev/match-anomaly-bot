"""Utilities for converting decimal odds to probabilities and back.

Using probabilities (instead of raw odds) removes bookmaker margin and allows
fair comparison across bookmakers with different overround levels.
"""
from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .odds_client import BookmakerOdds, MatchOdds


def implied_probability(odds: float) -> float:
    """Convert decimal odds to raw implied probability (1/odds)."""
    if odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got {odds}")
    return 1.0 / odds


def remove_overround(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities so they sum to 1.0, removing bookmaker margin."""
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("sum of probabilities must be positive")
    return {k: v / total for k, v in probs.items()}


def probabilities_from_match(bookmaker_odds: "BookmakerOdds") -> dict[str, float] | None:
    """Convert a single BookmakerOdds entry to margin-free {home,draw,away} probabilities.

    Returns None if fewer than 2 outcomes have valid prices.
    """
    raw: dict[str, float] = {}
    for outcome, price in (
        ("home", bookmaker_odds.home),
        ("draw", bookmaker_odds.draw),
        ("away", bookmaker_odds.away),
    ):
        if price is not None and price > 1.0:
            raw[outcome] = 1.0 / price

    if len(raw) < 2:
        return None

    return remove_overround(raw)


def consensus_probabilities(match: "MatchOdds") -> dict[str, float] | None:
    """Return median margin-free probabilities across all bookmakers.

    Median is more robust than mean — it's resistant to outlier bookmakers
    who haven't yet updated their lines.
    """
    by_outcome: dict[str, list[float]] = {"home": [], "draw": [], "away": []}

    for bm in match.bookmakers:
        probs = probabilities_from_match(bm)
        if probs is None:
            continue
        for outcome, p in probs.items():
            by_outcome[outcome].append(p)

    result: dict[str, float] = {}
    for outcome, values in by_outcome.items():
        if values:
            result[outcome] = statistics.median(values)

    return result if result else None


def probabilities_to_odds(probs: dict[str, float]) -> dict[str, float]:
    """Convert margin-free probabilities back to decimal odds for display."""
    return {k: 1.0 / v for k, v in probs.items() if v > 0}
