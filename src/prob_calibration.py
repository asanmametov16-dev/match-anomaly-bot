"""Калибровка консенсус-вероятностей против реальных исходов.

CLV меряет, двинулся ли рынок в сторону алерта. Этот модуль отвечает на
другой вопрос: **насколько вообще верны наши вероятности?** Берём
маржа-free consensus закрывающей линии (последний снимок до старта — та же
точка, что у CLV) и сравниваем с фактическим 1X2-исходом через Brier и
log-loss. Агрегат + кривая надёжности доступны командой /calibration.

Reference-точки для интерпретации Brier (многоклассовый, ∈ [0,2]):
- равномерный прогноз 1/3 на исход → Brier ≡ 0.667 (нет навыка);
- хороший рынок футбольного 1X2 → ≈ 0.55–0.58.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from .clv import _closing_snapshot, _reconstruct_match
from .db import OddsSnapshot, ProbCalibration, SessionLocal
from .probability import consensus_probabilities
from .result_checker import _fuzzy_find_result, _result_key

log = logging.getLogger(__name__)

_OUTCOMES = ("home", "draw", "away")
_POST_KICKOFF_BUFFER = timedelta(hours=2)
_GIVE_UP_HOURS = 96  # как в result_checker: дольше — лига вне football-data.org


def brier_score(probs: dict[str, float], actual: str) -> float:
    """Многоклассовый Brier: Σ (p_o − 1[o=факт])². 0 = идеально, 2 = худший."""
    return sum(
        (probs.get(o, 0.0) - (1.0 if o == actual else 0.0)) ** 2
        for o in _OUTCOMES
    )


def log_loss(probs: dict[str, float], actual: str) -> float:
    """−ln p(факт), клипован снизу — защита от log(0)."""
    return -math.log(max(probs.get(actual, 0.0), 1e-12))


def reliability_bins(
    points: list[tuple[float, int]], n_bins: int = 10
) -> list[dict]:
    """Кривая надёжности: точки (предсказанная p, попал ли 0/1) → бины.

    Каждый прогноз даёт 3 точки (по исходу). В калиброванном рынке средняя
    предсказанная вероятность ≈ эмпирической частоте попаданий внутри бина.
    Возвращает только непустые бины.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, hit in points:
        idx = min(int(p * n_bins), n_bins - 1)
        buckets[idx].append((p, hit))

    out: list[dict] = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        n = len(b)
        out.append({
            "lo": i / n_bins,
            "hi": (i + 1) / n_bins,
            "n": n,
            "mean_pred": sum(p for p, _ in b) / n,
            "emp_freq": sum(h for _, h in b) / n,
        })
    return out


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def compute_pending_calibration() -> int:
    """Оценить непросчитанные сыгранные матчи. Возвращает число новых строк.

    Идемпотентно: match_id уже в prob_calibration пропускается. Матч без
    результата/закрытия после 96ч пишется sentinel-строкой (NULL), чтобы не
    долбить fuzzy-поиск по непокрытым лигам вечно.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - _POST_KICKOFF_BUFFER
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    written = 0

    with SessionLocal() as session:
        done_subq = select(ProbCalibration.match_id)
        candidates = session.execute(
            select(
                OddsSnapshot.match_id,
                OddsSnapshot.home_team,
                OddsSnapshot.away_team,
                OddsSnapshot.commence_time,
            )
            .where(OddsSnapshot.commence_time <= cutoff)
            .where(OddsSnapshot.match_id.notin_(done_subq))
            .group_by(OddsSnapshot.match_id)
        ).all()

        for match_id, home, away, commence in candidates:
            age_h = (now_naive - commence.replace(tzinfo=None)).total_seconds() / 3600
            gave_up = age_h > _GIVE_UP_HOURS

            def _sentinel() -> None:
                session.add(ProbCalibration(match_id=match_id, actual=None,
                                            brier=None, log_loss=None))

            closing = _closing_snapshot(session, match_id, commence)
            probs = (
                consensus_probabilities(_reconstruct_match(closing))
                if closing is not None else None
            )
            if not probs or any(probs.get(o) is None for o in _OUTCOMES):
                if gave_up:
                    _sentinel()
                    written += 1
                continue  # 2-way рынок или нет закрытия — ждём/сдаёмся

            # consensus_probabilities — по-исходные взвешенные медианы, их
            # сумма ≈1, но не ровно. Нормируем, чтобы Brier/log-loss считались
            # на корректном распределении.
            tot = sum(probs[o] for o in _OUTCOMES)
            if tot <= 0:
                continue
            probs = {o: probs[o] / tot for o in _OUTCOMES}

            result = _fuzzy_find_result(session, home, away, commence)
            if result is None:
                if gave_up:
                    _sentinel()
                    written += 1
                continue

            actual = _actual_outcome(result.home_score, result.away_score)
            session.add(ProbCalibration(
                match_id=match_id,
                result_key=_result_key(home, away, commence),
                p_home=probs["home"], p_draw=probs["draw"], p_away=probs["away"],
                actual=actual,
                brier=brier_score(probs, actual),
                log_loss=log_loss(probs, actual),
            ))
            written += 1

        if written:
            session.commit()

    if written:
        log.info("Калибровка: добавлено %d матчей", written)
    return written


def calibration_summary() -> dict:
    """Агрегат для /calibration: n, средние Brier/log-loss, кривая надёжности."""
    with SessionLocal() as session:
        rows = session.execute(
            select(ProbCalibration).where(ProbCalibration.brier.isnot(None))
        ).scalars().all()
        pending = session.scalar(
            select(func.count(ProbCalibration.match_id))
            .where(ProbCalibration.brier.is_(None))
        ) or 0

    if not rows:
        return {"n": 0, "pending": pending}

    n = len(rows)
    points: list[tuple[float, int]] = []
    for r in rows:
        probs = {"home": r.p_home, "draw": r.p_draw, "away": r.p_away}
        for o in _OUTCOMES:
            points.append((probs[o], 1 if o == r.actual else 0))

    return {
        "n": n,
        "pending": pending,
        "mean_brier": sum(r.brier for r in rows) / n,
        "mean_log_loss": sum(r.log_loss for r in rows) / n,
        "uniform_brier": 2.0 / 3.0,  # ориентир «нет навыка»
        "bins": reliability_bins(points),
    }
