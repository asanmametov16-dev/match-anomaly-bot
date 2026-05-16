"""Тесты бэкфилла исторической калибровки модели sstats.

Сеть замокана httpx.MockTransport; async через asyncio.run. БД in-memory,
init_db застаблен (не трогаем реальный engine).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import src.sstats_client as sc
import src.sstats_history as sh
from src.db import Base, SstatsModelOutcome
from src.prob_calibration import brier_score
from src.sstats_history import (_actual, _parse_ended_item,
                                sstats_model_summary)


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    sc._xg_cache.clear()
    sc._daily_indexes.clear()
    monkeypatch.setattr(sc.settings, "sstats_enabled", True)
    monkeypatch.setattr(sc.settings, "sstats_api_key", "k")
    monkeypatch.setattr(sh.settings, "sstats_enabled", True)
    monkeypatch.setattr(sh.settings, "sstats_api_key", "k")
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(sh, "SessionLocal", Session)
    monkeypatch.setattr(sh, "init_db", lambda: None)
    yield Session
    sc._xg_cache.clear()


def _install(monkeypatch, handler):
    orig = httpx.AsyncClient.__init__

    def patched(self, *a, **k):
        k["transport"] = httpx.MockTransport(handler)
        orig(self, *a, **k)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


def _glicko(hwp, awp, hxg=1.4, axg=1.0):
    return {"status": "OK", "data": {"glicko": {
        "homeRating": 1500, "awayRating": 1480,
        "homeXg": hxg, "awayXg": axg,
        "homeWinProbability": hwp, "awayWinProbability": awp}}}


# --- чистая логика -----------------------------------------------------------

def test_actual():
    assert _actual(2, 0) == "home"
    assert _actual(0, 3) == "away"
    assert _actual(1, 1) == "draw"


def test_parse_alt_score_keys():
    g = {"id": 7, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
         "score": {"home": 1, "away": 2}, "league": {"name": "EPL"},
         "date": "2026-05-01T18:00:00Z"}
    p = _parse_ended_item(g)
    assert p["game_id"] == 7 and p["hs"] == 1 and p["as"] == 2
    assert p["league"] == "EPL" and p["played"] is not None


def test_parse_missing_score_returns_none():
    g = {"id": 7, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}}
    assert _parse_ended_item(g) is None


# --- бэкфилл end-to-end ------------------------------------------------------

def _handler(page1):
    def h(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/Games/list" in url:
            # первая страница (Offset=0) с матчами, дальше — пусто (стоп)
            if "Offset=0" in url:
                return httpx.Response(200, json={"data": page1})
            return httpx.Response(200, json={"data": []})
        if "/Games/glicko/1" in url:
            return httpx.Response(200, json=_glicko(0.55, 0.20))
        if "/Games/glicko/2" in url:
            return httpx.Response(200, json=_glicko(0.30, 0.45))
        if "/Games/glicko/3" in url:  # без winProb → пропуск
            return httpx.Response(200, json=_glicko(None, None))
        return httpx.Response(404, json={})
    return h


_PAGE = [
    {"id": 1, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
     "homeScore": 2, "awayScore": 0, "league": {"name": "EPL"}},
    {"id": 2, "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"},
     "homeScore": 0, "awayScore": 1, "league": {"name": "EPL"}},
    {"id": 3, "homeTeam": {"name": "E"}, "awayTeam": {"name": "F"},
     "homeScore": 1, "awayScore": 1, "league": {"name": "L2"}},
]


def test_backfill_writes_and_scores(reset, monkeypatch):
    _install(monkeypatch, _handler(_PAGE))
    written = asyncio.run(sh.backfill(max_games=10, page_limit=50, sleep=0))
    assert written == 2  # id=3 без winProb пропущен

    with reset() as s:
        rows = {r.game_id: r for r in
                s.execute(select(SstatsModelOutcome)).scalars()}
    assert set(rows) == {1, 2}
    r1 = rows[1]
    assert r1.actual == "home"
    probs = {"home": r1.p_home, "draw": r1.p_draw, "away": r1.p_away}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert r1.brier == pytest.approx(brier_score(probs, "home"))


def test_backfill_idempotent(reset, monkeypatch):
    _install(monkeypatch, _handler(_PAGE))
    assert asyncio.run(sh.backfill(max_games=10, page_limit=50, sleep=0)) == 2
    assert asyncio.run(sh.backfill(max_games=10, page_limit=50, sleep=0)) == 0


def test_summary_aggregates(reset, monkeypatch):
    _install(monkeypatch, _handler(_PAGE))
    asyncio.run(sh.backfill(max_games=10, page_limit=50, sleep=0))
    s = sstats_model_summary()
    assert s["n"] == 2
    assert 0.0 <= s["mean_brier"] <= 2.0
    assert s["uniform_brier"] == pytest.approx(2 / 3)
    assert s["leagues"][0]["league"] == "EPL" and s["leagues"][0]["n"] == 2


def test_summary_empty():
    assert sstats_model_summary() == {"n": 0}
