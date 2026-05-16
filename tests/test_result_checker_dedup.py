"""Регресс: дедуп MatchResult в пределах фетча (dual-source).

Раньше два матча с одинаковым нормализованным result_key (перекрытие
football-data ↔ sstats или коллизия имён) роняли батч UNIQUE constraint,
т.к. session.get не видит ещё-не-сфлашенные дубли. Сетевые фетчеры
замоканы; БД in-memory; notifier._bot=None → Telegram не трогается.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import src.result_checker as rc
import src.sstats_history as sh
from src.db import Base, MatchResult, ResultNotification
from src.results_client import FinishedMatch


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(rc, "SessionLocal", Session)
    return Session


def _fm(h, a, hs, as_):
    return FinishedMatch(home_team=h, away_team=a, home_score=hs,
                         away_score=as_,
                         utc_date=datetime(2026, 5, 15, 18, tzinfo=timezone.utc),
                         competition="L")


def _patch_sources(monkeypatch, fd_list, sstats_list):
    async def fd(days_back=3):
        return fd_list

    async def ss(days_back=3, max_pages=8):
        return sstats_list

    monkeypatch.setattr(rc, "fetch_finished_matches", fd)
    monkeypatch.setattr(sh, "fetch_finished_matches_sstats", ss)


def test_within_source_duplicate_collapses(db, monkeypatch):
    _patch_sources(monkeypatch, [_fm("A FC", "B FC", 2, 1),
                                 _fm("A FC", "B FC", 2, 1)], [])
    asyncio.run(rc.check_anomaly_results())
    with db() as s:
        assert s.scalar(select(func.count(MatchResult.result_key))) == 1


def test_cross_source_duplicate_collapses(db, monkeypatch):
    # тот же матч пришёл и из football-data, и из sstats
    _patch_sources(monkeypatch, [_fm("A FC", "B FC", 3, 0)],
                                [_fm("A FC", "B FC", 3, 0)])
    asyncio.run(rc.check_anomaly_results())
    with db() as s:
        assert s.scalar(select(func.count(MatchResult.result_key))) == 1


def test_distinct_matches_kept(db, monkeypatch):
    _patch_sources(monkeypatch, [_fm("A FC", "B FC", 1, 0)],
                                [_fm("C FC", "D FC", 2, 2)])
    asyncio.run(rc.check_anomaly_results())
    with db() as s:
        assert s.scalar(select(func.count(MatchResult.result_key))) == 2


def test_clear_false_sentinels_only(db):
    with db() as s:
        s.add_all([
            ResultNotification(result_key="k_false", result_found=False),
            ResultNotification(result_key="k_true", result_found=True),
            ResultNotification(result_key="k_none", result_found=None),
        ])
        s.commit()

    assert rc.clear_false_sentinels() == 1      # удалён только False
    assert rc.clear_false_sentinels() == 0      # идемпотентно
    with db() as s:
        keys = set(s.execute(
            select(ResultNotification.result_key)).scalars())
    assert keys == {"k_true", "k_none"}
