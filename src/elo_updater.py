"""Джоб обновления Elo-рейтингов по результатам сыгранных матчей.

Запускается раз в сутки. Помечает обработанные матчи в отдельной таблице,
чтобы не апдейтить рейтинги дважды.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import Column, DateTime, String

from .db import Base, SessionLocal, engine
from .elo import update_ratings
from .results_client import fetch_finished_matches, normalize_team_name

log = logging.getLogger(__name__)


class ProcessedResult(Base):
    """Маркер обработанного результата, чтобы не апдейтить Elo дважды."""
    __tablename__ = "processed_results"

    # Ключ: дата + нормализованные имена команд
    key = Column(String, primary_key=True)
    processed_at = Column(DateTime, default=datetime.utcnow)


# Создаём таблицу при импорте — db.init_db уже вызван в main.
Base.metadata.create_all(engine)


def _result_key(home: str, away: str, dt: datetime) -> str:
    return f"{dt.date().isoformat()}|{normalize_team_name(home)}|{normalize_team_name(away)}"


async def update_elo_from_results() -> None:
    """Подтягивает результаты за последние 2 дня и обновляет Elo."""
    matches = await fetch_finished_matches(days_back=2)
    if not matches:
        return

    updated = 0
    with SessionLocal() as session:
        for m in matches:
            key = _result_key(m.home_team, m.away_team, m.utc_date)
            if session.get(ProcessedResult, key) is not None:
                continue  # уже обработали

            # Имена в Elo-таблице храним как они приходят от Odds API,
            # поэтому нормализуем при поиске и при обновлении в elo.py
            # (это — точка для будущего улучшения с fuzzy-matching).
            update_ratings(
                session,
                home_team=normalize_team_name(m.home_team),
                away_team=normalize_team_name(m.away_team),
                home_score=m.home_score,
                away_score=m.away_score,
            )
            session.add(ProcessedResult(key=key))
            updated += 1

        session.commit()

    log.info("Elo обновлён по %d матчам", updated)
