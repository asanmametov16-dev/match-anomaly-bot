"""Utilities for converting decimal odds to probabilities and back.

Using probabilities (instead of raw odds) removes bookmaker margin and allows
fair comparison across bookmakers with different overround levels.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from .odds_client import BookmakerOdds, MatchOdds


def implied_probability(odds: float) -> float:
    """Convert decimal odds to raw implied probability (1/odds)."""
    if odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got {odds}")
    return 1.0 / odds


def _devig_proportional(probs: dict[str, float], total: float) -> dict[str, float]:
    """Простое пропорциональное снятие маржи: делим на сумму.

    Систематически смещено: маржа на самом деле концентрируется на
    аутсайдерах (favourite-longshot bias), а здесь снимается одинаковой
    долей со всех исходов — фавориты занижаются, аутсайдеры/ничьи
    завышаются. Оставлено как метод для отката (settings.devig_method).
    """
    return {k: v / total for k, v in probs.items()}


def _devig_shin(probs: dict[str, float], total: float) -> dict[str, float]:
    """Метод Шина: маржа моделируется как защита букмекера от инсайдеров.

    Доля «инсайдерских» денег z ∈ [0,1) подбирается так, чтобы истинные
    вероятности
        q_i = (sqrt(z² + 4(1−z)·p_i²/B) − z) / (2(1−z))
    суммировались в 1 (B = сумма implied prob = overround). Σq_i строго
    убывает по z, поэтому z находится бисекцией. Шин лучше пропорционального
    воспроизводит favourite-longshot bias — стандарт для футбольного 1X2.

    Вырожденные входы (нет маржи B≤1, неположительные prob) → откат на
    пропорциональный метод.
    """
    p = list(probs.values())
    if total <= 1.0 or any(v <= 0.0 for v in p):
        return _devig_proportional(probs, total)

    def q(z: float) -> list[float]:
        return [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * v * v / total) - z)
            / (2.0 * (1.0 - z))
            for v in p
        ]

    lo, hi = 0.0, 0.999  # Σq(0)=sqrt(B)>1, Σq убывает с ростом z
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sum(q(mid)) > 1.0:
            lo = mid
        else:
            hi = mid

    qs = q((lo + hi) / 2.0)
    norm = sum(qs)  # снимаем крошечный остаток бисекции
    return {k: v / norm for k, v in zip(probs.keys(), qs)}


def remove_overround(probs: dict[str, float]) -> dict[str, float]:
    """Снять маржу букмекера: вероятности нормируются в сумму 1.0.

    Метод выбирается settings.devig_method ∈ {"shin", "proportional"}.
    По умолчанию Shin — точнее для 1X2. Откат на "proportional" не требует
    пересчёта данных, только перезапуска.
    """
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("sum of probabilities must be positive")
    if settings.devig_method == "proportional":
        return _devig_proportional(probs, total)
    return _devig_shin(probs, total)


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


def bookmaker_weight(bookmaker_key: str) -> float:
    """Return consensus weight for a bookmaker key.

    Sharp bookmakers (Pinnacle, Betfair exchanges, etc.) price efficiently and
    move first — they deserve full weight. Soft bookmakers copy the line later
    and carry less information, so they get a lower weight.
    """
    sharp_set = {b.lower() for b in settings.sharp_bookmakers}
    if bookmaker_key.lower() in sharp_set:
        return settings.sharp_weight
    return settings.default_weight


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted median: more outlier-robust than weighted mean.

    Weighted mean lets a single high-weight outlier shift the result
    proportionally to its weight. Weighted median caps the outlier's
    influence: it can only pull the result to its own value, no further.
    This matters for bookmaker consensus — a stale soft book or an aggressive
    sharp open can't corrupt the consensus beyond their own price.
    """
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    cumulative = 0.0
    for val, w in paired:
        cumulative += w
        if cumulative >= total / 2:
            return val
    return paired[-1][0]


def consensus_probabilities(match: "MatchOdds") -> dict[str, float] | None:
    """Return weighted-median margin-free probabilities across all bookmakers.

    Weighted median (vs weighted mean) is chosen for outlier robustness:
    a single bookmaker cannot shift the consensus past its own value, so a
    stale soft-book line doesn't corrupt the signal.
    Sharp bookmakers receive weight=sharp_weight (1.0); others get
    default_weight (0.4) — see config.py.
    """
    by_outcome: dict[str, list[tuple[float, float]]] = {
        "home": [], "draw": [], "away": [],
    }

    for bm in match.bookmakers:
        probs = probabilities_from_match(bm)
        if probs is None:
            continue
        w = bookmaker_weight(bm.bookmaker)
        for outcome, p in probs.items():
            by_outcome[outcome].append((p, w))

    result: dict[str, float] = {}
    for outcome, pairs in by_outcome.items():
        if pairs:
            vals, weights = zip(*pairs)
            result[outcome] = _weighted_median(list(vals), list(weights))

    return result if result else None


def probabilities_to_odds(probs: dict[str, float]) -> dict[str, float]:
    """Convert margin-free probabilities back to decimal odds for display."""
    return {k: 1.0 / v for k, v in probs.items() if v > 0}
