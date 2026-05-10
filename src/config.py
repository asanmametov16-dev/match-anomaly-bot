"""Конфигурация приложения. Читается из .env через pydantic-settings."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    odds_api_key: str
    odds_api_regions: str = "eu,uk"
    odds_api_sport: str = "soccer"
    odds_api_markets: str = "h2h,totals,spreads"
    football_data_key: str = ""  # пусто = не обновлять Elo автоматически

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Расписание
    poll_interval_minutes: int = 15
    elo_update_hour_utc: int = 9  # ежедневно в 09:00 UTC (12:00 МСК)

    # Пороги детекторов
    spread_threshold: float = 0.08          # deprecated, kept for rollback
    spread_pp_threshold: float = 4.0        # порог в процентных пунктах (4.0pp = разница 50%→54%)
    drift_threshold: float = 0.15
    model_gap_threshold: float = 0.20
    sync_move_threshold: float = 0.05      # сдвиг у одного букмекера для счёта
    sync_min_bookmakers: int = 3           # минимум контор, двинувших одновременно
    exotic_spread_threshold: float = 0.10  # для тоталов и фор
    sharp_move_threshold: float = 0.05    # разрыв sharp vs soft (5%)

    # Фильтрация алертов
    alert_window_hours: int = 48       # алертить только матчи в ближайшие N часов
    alert_min_detectors: int = 2       # минимум детекторов (без exotic_spread) для алерта
    min_bookmakers_per_match: int = 6  # матчи с меньшим числом контор пропускаем целиком

    # Elo
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 60.0
    elo_default_rating: float = 1500.0

    # Прочее
    db_path: str = "data/anomalies.sqlite"
    log_level: str = "INFO"

    @property
    def db_url(self) -> str:
        path = Path(self.db_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"


settings = Settings()
