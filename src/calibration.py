"""Калибровка весов детекторов по накопленному CLV.

`compute_score` исторически суммировал захардкоженные `DETECTOR_WEIGHTS`
(на глаз). CLV (closing line value, см. clv.py) — объективная мера
сигнальной ценности детектора: если рынок систематически НЕ двигался
в сторону его срабатываний, детектор переоценён.

Этот модуль раз в час пересчитывает по таблице `AnomalyCLV` средний CLV
на детектор и превращает его в множитель к базовому весу:

    multiplier = clamp(1 + sensitivity * mean_clv_pp, min_mult, max_mult)

Пока по детектору меньше `clv_calibration_min_samples` измерений с
ненулевым clv_pp — множитель = 1.0 (доверяем ручному дефолту, не шумим
на 3 точках). Результат кэшируется в модуле; `detector_multiplier`
читается из горячего пути детекторов без обращения к БД.
"""
from __future__ import annotations

import logging

from sqlalchemy import func, select

from .config import settings
from .db import AnomalyCLV, SessionLocal

log = logging.getLogger(__name__)

# Горячий кэш: detector -> множитель. Пустой = все детекторы нейтральны (1.0).
_multipliers: dict[str, float] = {}
# Для прозрачности в /weights: detector -> (n, mean_clv_pp).
_stats: dict[str, tuple[int, float]] = {}


def multiplier_for_mean_clv(mean_clv_pp: float) -> float:
    """Средний CLV (в процентных пунктах) → ограниченный множитель веса."""
    raw = 1.0 + settings.clv_calibration_sensitivity * mean_clv_pp
    return max(
        settings.clv_calibration_min_multiplier,
        min(settings.clv_calibration_max_multiplier, raw),
    )


def refresh_detector_weights() -> dict[str, float]:
    """Пересчитать множители из AnomalyCLV и обновить кэш. Возвращает кэш.

    Вызывается шедулером ежечасно и один раз на старте. Детекторы с числом
    измерений < порога в кэш не попадают → detector_multiplier вернёт 1.0.
    """
    new_mult: dict[str, float] = {}
    new_stats: dict[str, tuple[int, float]] = {}

    with SessionLocal() as session:
        rows = session.execute(
            select(
                AnomalyCLV.detector,
                func.count(AnomalyCLV.anomaly_id),
                func.avg(AnomalyCLV.clv_pp),
            )
            .where(AnomalyCLV.clv_pp.isnot(None))
            .group_by(AnomalyCLV.detector)
        ).all()

    for detector, n, mean_clv in rows:
        n = int(n or 0)
        mean_clv = float(mean_clv if mean_clv is not None else 0.0)
        new_stats[detector] = (n, mean_clv)
        if n >= settings.clv_calibration_min_samples:
            new_mult[detector] = multiplier_for_mean_clv(mean_clv)

    _multipliers.clear()
    _multipliers.update(new_mult)
    _stats.clear()
    _stats.update(new_stats)

    if new_mult:
        pretty = ", ".join(f"{d}×{m:.2f}" for d, m in sorted(new_mult.items()))
        log.info("CLV-калибровка весов: %s", pretty)
    else:
        log.info("CLV-калибровка: пока недостаточно данных, веса дефолтные")
    return dict(_multipliers)


def detector_multiplier(detector: str) -> float:
    """Множитель веса детектора. 1.0, если калибровка off или мало данных."""
    if not settings.clv_calibration_enabled:
        return 1.0
    return _multipliers.get(detector, 1.0)


def current_calibration() -> dict[str, tuple[int, float, float]]:
    """Снимок для /weights: detector -> (n, mean_clv_pp, эффективный множитель)."""
    out: dict[str, tuple[int, float, float]] = {}
    for detector, (n, mean_clv) in _stats.items():
        out[detector] = (n, mean_clv, detector_multiplier(detector))
    return out
