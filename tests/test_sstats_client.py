"""Тесты для src/sstats_client.py — обогащение xG/Glicko для model_gap.

Сетевые вызовы замоканы через httpx.MockTransport. Async-тесты обёрнуты
в asyncio.run() — без зависимости от pytest-asyncio.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src import sstats_client
from src.odds_client import MatchOdds, BookmakerOdds


def _make_match(home: str, away: str, kickoff_utc: datetime, mid: str = "m1") -> MatchOdds:
    return MatchOdds(
        match_id=mid,
        sport_key="soccer_test",
        home_team=home,
        away_team=away,
        commence_time=kickoff_utc,
        bookmakers=[BookmakerOdds(bookmaker="bm", home=2.0, draw=3.5, away=4.0)],
    )


@pytest.fixture(autouse=True)
def reset_caches():
    """Каждый тест — чистый кэш на уровне модуля."""
    sstats_client._daily_indexes.clear()
    sstats_client._xg_cache.clear()
    yield
    sstats_client._daily_indexes.clear()
    sstats_client._xg_cache.clear()


@pytest.fixture
def settings_with_key(monkeypatch):
    """Включает sstats и подсовывает фейк-ключ."""
    monkeypatch.setattr(sstats_client.settings, "sstats_enabled", True)
    monkeypatch.setattr(sstats_client.settings, "sstats_api_key", "test_key")


def _install_mock_transport(monkeypatch, handler):
    """Подменяет httpx.AsyncClient так, чтобы он использовал MockTransport."""
    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


# ---------------------------------------------------------------------------
# Guards: disabled / no-key — мгновенный empty result, без HTTP
# ---------------------------------------------------------------------------

def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(sstats_client.settings, "sstats_enabled", False)
    monkeypatch.setattr(sstats_client.settings, "sstats_api_key", "any")
    match = _make_match("A", "B", datetime.now(timezone.utc) + timedelta(hours=24))
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert out == {}


def test_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr(sstats_client.settings, "sstats_enabled", True)
    monkeypatch.setattr(sstats_client.settings, "sstats_api_key", "")
    match = _make_match("A", "B", datetime.now(timezone.utc) + timedelta(hours=24))
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert out == {}


def test_empty_matches_list(settings_with_key):
    out = asyncio.run(sstats_client.fetch_xg_batch([]))
    assert out == {}


# ---------------------------------------------------------------------------
# Full path: team-matching + xG → возвращается XgPrediction
# ---------------------------------------------------------------------------

def test_fetch_xg_batch_matches_and_returns_prediction(monkeypatch, settings_with_key):
    kickoff = datetime(2026, 5, 15, 18, 0, tzinfo=timezone.utc)
    date_str = "2026-05-15"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/Games/list" in url and f"Date={date_str}" in url:
            return httpx.Response(200, json={
                "status": "OK", "count": 1, "offset": 0,
                "data": [{
                    "id": 999,
                    "homeTeam": {"name": "Celta Vigo"},
                    "awayTeam": {"name": "Levante"},
                }],
            })
        if "/Games/glicko/999" in url:
            return httpx.Response(200, json={
                "status": "OK",
                "data": {"glicko": {
                    "homeRating": 1548.0, "awayRating": 1462.0,
                    "homeXg": 1.58, "awayXg": 1.03,
                    "homeWinProbability": 0.50, "awayWinProbability": 0.25,
                }},
            })
        return httpx.Response(404, json={})

    _install_mock_transport(monkeypatch, handler)
    match = _make_match("Celta Vigo", "Levante", kickoff)
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))

    assert "m1" in out
    pred = out["m1"]
    assert pred.home_xg == pytest.approx(1.58)
    assert pred.away_xg == pytest.approx(1.03)
    assert pred.home_win_prob == pytest.approx(0.50)
    assert pred.away_win_prob == pytest.approx(0.25)
    # draw_prob выводится из 1 - home - away
    assert pred.draw_prob == pytest.approx(0.25)
    assert pred.home_glicko == pytest.approx(1548.0)


# ---------------------------------------------------------------------------
# xG/winProb null (низкие лиги) → не попадает в результат
# ---------------------------------------------------------------------------

def test_null_xg_excluded(monkeypatch, settings_with_key):
    kickoff = datetime(2026, 5, 15, 18, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/Games/list" in url:
            return httpx.Response(200, json={
                "data": [{
                    "id": 555,
                    "homeTeam": {"name": "Lower"}, "awayTeam": {"name": "Tier"},
                }],
            })
        if "/Games/glicko/555" in url:
            return httpx.Response(200, json={
                "data": {"glicko": {
                    "homeRating": 1500.0, "awayRating": 1500.0,
                    "homeXg": None, "awayXg": None,
                    "homeWinProbability": None, "awayWinProbability": None,
                }},
            })
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, handler)
    match = _make_match("Lower", "Tier", kickoff)
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert out == {}, "Матчи без xG не должны попадать в результат"


# ---------------------------------------------------------------------------
# Team name normalization работает (Manchester United vs Man United и т.п.)
# ---------------------------------------------------------------------------

def test_team_name_normalization(monkeypatch, settings_with_key):
    kickoff = datetime(2026, 5, 15, 18, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/Games/list" in url:
            return httpx.Response(200, json={
                "data": [{
                    "id": 100,
                    "homeTeam": {"name": "Arsenal FC"},   # sstats с суффиксом FC
                    "awayTeam": {"name": "Chelsea"},
                }],
            })
        if "/Games/glicko/100" in url:
            return httpx.Response(200, json={
                "data": {"glicko": {
                    "homeRating": 1600, "awayRating": 1580,
                    "homeXg": 1.4, "awayXg": 1.2,
                    "homeWinProbability": 0.42, "awayWinProbability": 0.30,
                }},
            })
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, handler)
    # В Odds API имя без FC — нормализация должна совпасть с sstats's "Arsenal FC"
    match = _make_match("Arsenal", "Chelsea", kickoff)
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert "m1" in out
    assert out["m1"].home_xg == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Кэш: повторный вызов не делает повторного HTTP-запроса
# ---------------------------------------------------------------------------

def test_cache_avoids_repeated_requests(monkeypatch, settings_with_key):
    call_count = {"list": 0, "glicko": 0}
    kickoff = datetime(2026, 5, 15, 18, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/Games/list" in url:
            call_count["list"] += 1
            return httpx.Response(200, json={
                "data": [{"id": 1, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}}],
            })
        if "/Games/glicko/1" in url:
            call_count["glicko"] += 1
            return httpx.Response(200, json={
                "data": {"glicko": {
                    "homeRating": 1500, "awayRating": 1500,
                    "homeXg": 1.0, "awayXg": 1.0,
                    "homeWinProbability": 0.4, "awayWinProbability": 0.3,
                }},
            })
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, handler)
    match = _make_match("A", "B", kickoff)

    out1 = asyncio.run(sstats_client.fetch_xg_batch([match]))
    out2 = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert "m1" in out1 and "m1" in out2
    assert call_count == {"list": 1, "glicko": 1}, \
        f"повторный вызов должен использовать кэш, а сделал {call_count}"


# ---------------------------------------------------------------------------
# Ошибка HTTP не валит pipeline — возвращается пустой dict (или меньше матчей)
# ---------------------------------------------------------------------------

def test_http_500_swallowed(monkeypatch, settings_with_key):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server failure"})

    _install_mock_transport(monkeypatch, handler)
    match = _make_match("A", "B", datetime.now(timezone.utc) + timedelta(hours=24))
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert out == {}, "HTTP 500 не должен валить функцию"


def test_team_not_found_excluded(monkeypatch, settings_with_key):
    """Матч есть в нашем pipeline, но в sstats его нет — корректно пропускается."""
    kickoff = datetime(2026, 5, 15, 18, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/Games/list" in url:
            return httpx.Response(200, json={
                "data": [{"id": 1, "homeTeam": {"name": "Other"}, "awayTeam": {"name": "Match"}}],
            })
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, handler)
    match = _make_match("Unknown", "Team", kickoff)
    out = asyncio.run(sstats_client.fetch_xg_batch([match]))
    assert out == {}
