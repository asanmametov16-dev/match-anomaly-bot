"""Обслуживание БД: выборочный сброс detector-зависимых данных.

Зачем: после смены de-vig (Shin), порогов, набора детекторов и
precision-gate старые Anomaly/OddsSnapshot/CLV/калибровка сгенерены
ДРУГОЙ логикой — мешать их с новыми при калибровке #1/#4 статистически
грязно. Чистим только то, что зависит от логики детекторов.

НЕ трогаем (не зависит от наших детекторов, дорого/незачем терять):
- SstatsModelOutcome — калибровка модели sstats (фундамент #3, бэкфилл),
- MatchResult        — фактические результаты матчей (reference),
- TeamRating, ProcessedResult — Elo и его дедуп-маркеры.
"""
from __future__ import annotations

import logging

from .db import (Anomaly, AnomalyCLV, AnomalyOutcome, OddsSnapshot,
                 ProbCalibration, ResultNotification, SessionLocal)

log = logging.getLogger(__name__)

# Порядок: дочерние/зависимые раньше (FK тут нет, но порядок осмысленный).
_RESET_TABLES = [
    ("anomaly_clv", AnomalyCLV),
    ("anomaly_outcomes", AnomalyOutcome),
    ("prob_calibration", ProbCalibration),
    ("result_notifications", ResultNotification),
    ("anomalies", Anomaly),
    ("odds_snapshots", OddsSnapshot),
]


def purge_detector_data() -> dict[str, int]:
    """Удалить все строки detector-зависимых таблиц. Возвращает {table: n}.

    Идемпотентно (повторный вызов вернёт нули). Не затрагивает
    SstatsModelOutcome / MatchResult / TeamRating / ProcessedResult.
    """
    deleted: dict[str, int] = {}
    with SessionLocal() as session:
        for name, model in _RESET_TABLES:
            n = session.query(model).delete(synchronize_session=False)
            deleted[name] = int(n or 0)
        session.commit()
    log.info("purge_detector_data: %s", deleted)
    return deleted
