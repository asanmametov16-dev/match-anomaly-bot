"""SQLAlchemy-модели и инициализация БД."""
from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


class OddsSnapshot(Base):
    """Снимок коэффициентов на матч в момент времени.

    Один матч = много снимков (по одному на каждый poll), что позволяет
    детектору 'drift' видеть движение линии.
    """
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True, nullable=False)  # id из API
    sport_key = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    commence_time = Column(DateTime, nullable=False, index=True)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Сводные значения (медиана по букмекерам)
    median_home = Column(Float)
    median_draw = Column(Float)
    median_away = Column(Float)

    # Полные данные по всем букмекерам — JSON: [{"bookmaker": "...", "home":..,"draw":..,"away":..}, ...]
    bookmakers = Column(JSON, nullable=False)


class TeamRating(Base):
    """Текущий Elo-рейтинг команды."""
    __tablename__ = "team_ratings"

    team = Column(String, primary_key=True)
    rating = Column(Float, nullable=False)
    games_played = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnomalyOutcome(Base):
    """Вердикт по каждому детектору после завершения матча."""
    __tablename__ = "anomaly_outcomes"

    id = Column(Integer, primary_key=True)
    result_key = Column(String, nullable=False, index=True)
    detector = Column(String, nullable=False)
    # 1 = подтвердилась, 0 = не подтвердилась, NULL = нет направления (spread/exotic)
    confirmed = Column(Integer, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class MatchResult(Base):
    """Финальный счёт матча, загруженный из football-data.org."""
    __tablename__ = "match_results"

    result_key = Column(String, primary_key=True)  # date|home_norm|away_norm
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)
    competition = Column(String)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class ResultNotification(Base):
    """Отслеживает, по каким матчам уже отправили результат в Telegram."""
    __tablename__ = "result_notifications"

    result_key = Column(String, primary_key=True)
    sent_at = Column(DateTime, default=datetime.utcnow)
    # True = результат найден и отправлен; False = лига вне football-data.org
    result_found = Column(Boolean, nullable=True, default=True)


class Anomaly(Base):
    """Зафиксированная аномалия — то, что отправляется в Telegram."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    commence_time = Column(DateTime, nullable=False)

    detector = Column(String, nullable=False)  # spread / drift / model_gap
    severity = Column(Float, nullable=False)   # числовая мера срабатывания
    details = Column(Text)                     # человекочитаемое описание
    payload = Column(JSON)                     # сырые данные для разбора


engine = create_engine(settings.db_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    # Миграция: добавляем колонку result_found если её ещё нет (старая БД)
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE result_notifications ADD COLUMN result_found BOOLEAN"
            ))
            conn.commit()
        except Exception:
            pass  # колонка уже есть
