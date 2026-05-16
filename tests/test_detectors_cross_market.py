"""Тесты detect_cross_market — тождество DNB между 1X2 и форой 0.0.

Синтетика, без сети/БД. Ожидаемые значения считаются теми же функциями
(consensus_probabilities / remove_overround), а не хардкодом — тест
проверяет поведение, а не конкретные числа de-vig.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.clv import extract_bet_side
from src.detectors import detect_cross_market
from src.odds_client import BookmakerOdds, MatchOdds
from src.probability import consensus_probabilities


def _ah0(home_team: str, away_team: str, q_home: float):
    """Фора 0.0 с нулевой маржой, дающая implied home-prob = q_home."""
    return [
        {"name": home_team, "point": 0.0, "price": 1.0 / q_home},
        {"name": away_team, "point": 0.0, "price": 1.0 / (1.0 - q_home)},
    ]


def _book(name, h, d, a, spreads=None):
    return BookmakerOdds(bookmaker=name, home=h, draw=d, away=a, spreads=spreads)


def _match(books, hours_until=48.0):
    return MatchOdds(
        match_id="m1", sport_key="soccer_test",
        home_team="Home FC", away_team="Away FC",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=hours_until),
        bookmakers=books,
    )


def _dnb_h2h(match) -> float:
    p = consensus_probabilities(match)
    return p["home"] / (p["home"] + p["away"])


def test_consistent_markets_no_hit():
    """Фора 0.0 ровно по DNB из 1X2 → расхождения нет."""
    base = _match([_book(f"bk{i}", 2.0, 3.5, 4.0) for i in range(3)])
    q = _dnb_h2h(base)
    books = [_book(f"bk{i}", 2.0, 3.5, 4.0, _ah0("Home FC", "Away FC", q))
             for i in range(3)]
    assert detect_cross_market(_match(books)) == []


def test_inconsistent_markets_fire():
    base = _match([_book(f"bk{i}", 2.0, 3.5, 4.0) for i in range(3)])
    q = _dnb_h2h(base) - 0.10  # фора 0.0 на 10пп расходится с 1X2
    books = [_book(f"bk{i}", 2.0, 3.5, 4.0, _ah0("Home FC", "Away FC", q))
             for i in range(3)]
    hits = detect_cross_market(_match(books))
    assert len(hits) == 1
    h = hits[0]
    assert h.detector == "cross_market"
    assert h.severity == pytest.approx(h.payload["gap_pp"])
    assert h.payload["gap_pp"] == pytest.approx(10.0, abs=0.5)
    assert h.payload["ah_books"] == 3


def test_non_directional_payload():
    """cross_market не направленный → CLV.extract_bet_side вернёт None."""
    base = _match([_book(f"bk{i}", 2.0, 3.5, 4.0) for i in range(3)])
    q = _dnb_h2h(base) - 0.12
    books = [_book(f"bk{i}", 2.0, 3.5, 4.0, _ah0("Home FC", "Away FC", q))
             for i in range(3)]
    h = detect_cross_market(_match(books))[0]
    assert "outcome" not in h.payload
    assert extract_bet_side("cross_market", h.payload) is None


def test_noop_when_too_few_ah_books():
    base = _match([_book(f"bk{i}", 2.0, 3.5, 4.0) for i in range(3)])
    q = _dnb_h2h(base) - 0.20
    # только 2 конторы с форой 0.0 (< min 3)
    books = [
        _book("bk0", 2.0, 3.5, 4.0, _ah0("Home FC", "Away FC", q)),
        _book("bk1", 2.0, 3.5, 4.0, _ah0("Home FC", "Away FC", q)),
        _book("bk2", 2.0, 3.5, 4.0),  # без форы
    ]
    assert detect_cross_market(_match(books)) == []


def test_noop_when_no_h2h_consensus():
    # ни одной валидной 1X2-цены → консенсус не считается
    books = [_book(f"bk{i}", None, None, None,
                   _ah0("Home FC", "Away FC", 0.5)) for i in range(3)]
    assert detect_cross_market(_match(books)) == []


def test_ignores_non_zero_handicap_lines():
    """Линии ≠ 0.0 не должны участвовать — тогда AH-книг < min → no-op."""
    base = _match([_book(f"bk{i}", 2.0, 3.5, 4.0) for i in range(3)])
    q = _dnb_h2h(base) - 0.15
    spreads = [
        {"name": "Home FC", "point": -0.5, "price": 1.0 / q},
        {"name": "Away FC", "point": 0.5, "price": 1.0 / (1.0 - q)},
    ]
    books = [_book(f"bk{i}", 2.0, 3.5, 4.0, spreads) for i in range(3)]
    assert detect_cross_market(_match(books)) == []
