"""Tests for src/elo_bootstrap.py — mocked HTTP, in-memory SQLite."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base, TeamRating
from src.elo_bootstrap import fetch_and_load

SAMPLE_CSV = """\
Rank,Club,Country,Level,Elo,From,To
1,Arsenal,ENG,1,2010,2026-01-01,2026-05-10
2,Bayern Munich,GER,1,1980,2026-01-01,2026-05-10
3,Real Madrid FC,ESP,1,2050,2026-01-01,2026-05-10
"""


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)


def _mock_response(text: str):
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


# --- Basic loading -----------------------------------------------------------

def test_loads_all_rows(session_factory):
    """Three valid CSV rows → three TeamRating entries."""
    with patch("src.elo_bootstrap.httpx.get", return_value=_mock_response(SAMPLE_CSV)):
        count = fetch_and_load(date(2026, 5, 10), session_factory=session_factory)

    assert count == 3
    with session_factory() as s:
        ratings = s.query(TeamRating).all()
    assert len(ratings) == 3


def test_ratings_stored_correctly(session_factory):
    """Elo values are stored under normalized team names."""
    with patch("src.elo_bootstrap.httpx.get", return_value=_mock_response(SAMPLE_CSV)):
        fetch_and_load(date(2026, 5, 10), session_factory=session_factory)

    with session_factory() as s:
        arsenal = s.get(TeamRating, "arsenal")
        assert arsenal is not None
        assert abs(arsenal.rating - 2010) < 0.01

        # "Real Madrid FC" → _normalize strips " FC" → "real madrid"
        real = s.get(TeamRating, "real madrid")
        assert real is not None
        assert abs(real.rating - 2050) < 0.01


# --- Upsert: existing records overwritten ------------------------------------

def test_overwrites_existing_rating(session_factory):
    """If a team already has a rating, it gets overwritten, not duplicated."""
    with session_factory() as s:
        s.add(TeamRating(team="arsenal", rating=1500, games_played=10))
        s.commit()

    with patch("src.elo_bootstrap.httpx.get", return_value=_mock_response(SAMPLE_CSV)):
        fetch_and_load(date(2026, 5, 10), session_factory=session_factory)

    with session_factory() as s:
        arsenal = s.get(TeamRating, "arsenal")
        assert abs(arsenal.rating - 2010) < 0.01  # updated from 1500
        # games_played is not reset by bootstrap
        assert arsenal.games_played == 10


# --- Bad rows are skipped gracefully -----------------------------------------

def test_skips_rows_with_missing_data(session_factory):
    """Rows with empty Club or Elo are ignored, rest are loaded."""
    csv_with_bad = (
        "Rank,Club,Country,Level,Elo,From,To\n"
        "1,Arsenal,ENG,1,2010,2026-01-01,2026-05-10\n"
        "2,,ENG,1,1900,2026-01-01,2026-05-10\n"        # empty club
        "3,Chelsea,ENG,1,,2026-01-01,2026-05-10\n"    # empty elo
        "4,Liverpool,ENG,1,1950,2026-01-01,2026-05-10\n"
    )
    with patch("src.elo_bootstrap.httpx.get", return_value=_mock_response(csv_with_bad)):
        count = fetch_and_load(date(2026, 5, 10), session_factory=session_factory)

    assert count == 2  # only Arsenal and Liverpool


def test_skips_rows_with_invalid_elo(session_factory):
    """Non-numeric Elo value is skipped without crashing."""
    csv_bad_elo = (
        "Rank,Club,Country,Level,Elo,From,To\n"
        "1,Arsenal,ENG,1,2010,2026-01-01,2026-05-10\n"
        "2,Unknown FC,ENG,1,N/A,2026-01-01,2026-05-10\n"
    )
    with patch("src.elo_bootstrap.httpx.get", return_value=_mock_response(csv_bad_elo)):
        count = fetch_and_load(date(2026, 5, 10), session_factory=session_factory)

    assert count == 1  # only Arsenal


# --- HTTP error is propagated ------------------------------------------------

def test_http_error_propagates(session_factory):
    """HTTPStatusError from clubelo.com bubbles up to the caller."""
    resp = _mock_response("")
    resp.raise_for_status.side_effect = Exception("HTTP 404")

    with patch("src.elo_bootstrap.httpx.get", return_value=resp):
        with pytest.raises(Exception, match="HTTP 404"):
            fetch_and_load(date(2026, 5, 10), session_factory=session_factory)
