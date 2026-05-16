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
    sstats_api_key: str = ""     # пусто = sstats обогащение выключено
    sstats_enabled: bool = True  # явный switch на случай аварийного откл.

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Расписание
    poll_interval_minutes: int = 15
    elo_update_hour_utc: int = 9  # ежедневно в 09:00 UTC (12:00 МСК)

    # Пороги детекторов
    spread_threshold: float = 0.08          # deprecated, kept for rollback
    spread_pp_threshold: float = 4.0        # порог в процентных пунктах (4.0pp = разница 50%→54%)
    drift_threshold: float = 0.15           # deprecated, kept for rollback
    drift_window_minutes: int = 120         # сравниваем с самым старым снимком в этом окне
    drift_pp_threshold: float = 5.0         # порог движения в процентных пунктах
    model_gap_threshold: float = 0.20
    sync_move_threshold: float = 0.05      # сдвиг у одного букмекера для счёта
    sync_min_bookmakers: int = 3           # минимум контор, двинувших одновременно
    exotic_spread_threshold: float = 0.10  # для тоталов и фор
    sharp_move_threshold: float = 0.05    # разрыв sharp vs soft (5%)

    # Веса букмекеров: sharp-конторы первыми двигают рынок и отражают «умные деньги»
    sharp_bookmakers: list[str] = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk",
                                   "sbobet", "matchbook"]
    sharp_weight: float = 1.0    # вес sharp-конторы в consensus_probabilities
    default_weight: float = 0.4  # вес остальных контор

    # Метод снятия маржи букмекера (см. probability.remove_overround)
    devig_method: str = "shin"  # "shin" (точнее) | "proportional" (для отката)

    # Калибровка весов детекторов по накопленному CLV (см. calibration.py)
    clv_calibration_enabled: bool = True
    clv_calibration_min_samples: int = 30      # меньше — вес детектора дефолтный
    clv_calibration_sensitivity: float = 0.15  # прибавка к множителю на 1пп ср. CLV
    clv_calibration_min_multiplier: float = 0.3
    clv_calibration_max_multiplier: float = 1.6

    # Временны́е корзины: порог × multiplier — чем ближе к матчу, тем чувствительнее
    # 0-6ч: ×0.70 (ловим больше), 6-24ч: ×0.85, 24-72ч: ×1.0 (база), 72ч+: ×1.3
    time_buckets_hours: list[int] = [6, 24, 72]
    time_bucket_multipliers: list[float] = [0.7, 0.85, 1.0, 1.3]

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
