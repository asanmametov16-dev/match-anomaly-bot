"""Bootstrap Elo ratings from clubelo.com.

clubelo.com provides a free daily CSV snapshot of Elo ratings for ~2000 clubs.
Run this once before starting the bot so that detect_model_gap works immediately
instead of needing 1-2 weeks to accumulate ratings from match results.

Usage:
    python -m src.elo_bootstrap            # uses today's date
    python -m src.elo_bootstrap 2026-05-10 # uses a specific date
"""
from __future__ import annotations

import csv
import io
import logging
import sys
from datetime import date
from typing import Callable

import httpx

from .db import SessionLocal, TeamRating, init_db
from .elo import _normalize

log = logging.getLogger(__name__)

CLUBELO_API = "http://api.clubelo.com/{date}/"


def fetch_and_load(
    target_date: date | None = None,
    session_factory: Callable | None = None,
) -> int:
    """Download clubelo.com ratings for target_date and upsert into TeamRating.

    Returns the number of team rows written.
    Existing records are overwritten; new ones are inserted.
    """
    if target_date is None:
        target_date = date.today()
    if session_factory is None:
        session_factory = SessionLocal

    url = CLUBELO_API.format(date=target_date.isoformat())
    log.info("Загружаем Elo с %s", url)

    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text))

    loaded = 0
    skipped = 0
    with session_factory() as session:
        for row in reader:
            club = (row.get("Club") or "").strip()
            elo_raw = (row.get("Elo") or "").strip()
            if not club or not elo_raw:
                skipped += 1
                continue
            try:
                elo = float(elo_raw)
            except ValueError:
                log.warning("Не удалось распарсить Elo для '%s': '%s'", club, elo_raw)
                skipped += 1
                continue

            key = _normalize(club)
            existing = session.get(TeamRating, key)
            if existing is not None:
                existing.rating = elo
            else:
                session.add(TeamRating(team=key, rating=elo, games_played=0))
            loaded += 1

        session.commit()

    log.info("Elo: загружено %d команд, пропущено %d строк", loaded, skipped)
    return loaded


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    init_db()

    target: date | None = None
    if len(sys.argv) > 1:
        try:
            target = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"Неверный формат даты: '{sys.argv[1]}'. Ожидается YYYY-MM-DD.")
            sys.exit(1)

    count = fetch_and_load(target)
    print(f"Готово: загружено {count} команд.")
