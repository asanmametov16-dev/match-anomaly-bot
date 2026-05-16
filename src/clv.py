"""CLV (Closing Line Value) — измерение прогностической ценности алертов.

Для каждой направленной аномалии (drift, synchronized, model_gap, sharp_move)
сравниваем взвешенный консенсус по «ставочной стороне» в момент алерта
с консенсусом в последнем снимке до старта матча. Положительный CLV в
процентных пунктах означает, что рынок двинулся дальше в сторону, на которую
указывал детектор — единственный честный способ проверить, есть ли у бота
прогностический сигнал.

Spread и exotic_spread детектируют расхождение, а не движение — у них нет
направленной гипотезы, поэтому CLV для них не считается.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Anomaly, AnomalyCLV, OddsSnapshot, SessionLocal
from .odds_client import BookmakerOdds, MatchOdds
from .probability import consensus_probabilities

log = logging.getLogger(__name__)

# Сколько ждать после старта матча перед расчётом CLV.
# Закладываем на случай, если последний poll прошёл за пару минут до старта.
_POST_KICKOFF_BUFFER = timedelta(minutes=5)


def extract_bet_side(detector: str, payload: dict | None) -> str | None:
    """Определить «ставочную сторону» (home/draw/away) по payload алерта.

    Возвращает None, если детектор не направленный (spread, exotic_spread),
    или направление неоднозначно (drift вниз — мы знаем, что эта сторона
    подешевела, но не знаем, куда перетекла вероятность). В таких случаях
    CLV не считается, и строка пишется с NULL-полями.
    """
    if not payload:
        return None
    outcome = payload.get("outcome")
    if outcome not in ("home", "draw", "away"):
        return None

    if detector == "drift":
        # drift_pp > 0: эта сторона подорожала в вероятности → бет на неё.
        # drift_pp < 0: эта сторона подешевела → бет «не на неё», но какая
        # из двух оставшихся? Пропускаем.
        return outcome if payload.get("drift_pp", 0) > 0 else None

    if detector == "synchronized":
        # direction "↓" = коэф. вниз = вероятность вверх = sharp-деньги на этой стороне.
        # direction "↑" — зеркальная неоднозначность, как у drift вниз.
        return outcome if payload.get("direction") == "↓" else None

    if detector == "model_gap":
        # market > fair → рынок недооценивает эту сторону → value-bet на outcome.
        market = payload.get("market")
        fair = payload.get("fair")
        if market is None or fair is None:
            return None
        return outcome if market > fair else None

    if detector == "sharp_move":
        # Детектор срабатывает только когда sharp > soft на этой стороне,
        # т.е. он по построению направленный — бет на outcome.
        return outcome

    # spread, exotic_spread, неизвестные — не направленные.
    return None


def _reconstruct_match(snap: OddsSnapshot) -> MatchOdds:
    """Восстановить MatchOdds из JSON-снимка для расчёта consensus_probabilities."""
    bms = [
        BookmakerOdds(
            bookmaker=b.get("bookmaker", ""),
            home=b.get("home"),
            draw=b.get("draw"),
            away=b.get("away"),
        )
        for b in (snap.bookmakers or [])
    ]
    return MatchOdds(
        match_id=snap.match_id,
        sport_key=snap.sport_key,
        home_team=snap.home_team,
        away_team=snap.away_team,
        commence_time=snap.commence_time,
        bookmakers=bms,
    )


def _alert_snapshot(session: Session, match_id: str,
                    detected_at: datetime) -> OddsSnapshot | None:
    """Снимок, сохранённый в том же цикле опроса, что и алерт.

    Pipeline пишет снимок сразу после детекторов в одном commit, поэтому
    captured_at ≈ detected_at (отличаются на миллисекунды). Берём
    ближайший снимок с captured_at <= detected_at + 1 мин — это та самая
    линия, которую видел детектор.
    """
    cutoff = detected_at + timedelta(minutes=1)
    return session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match_id)
        .where(OddsSnapshot.captured_at <= cutoff)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _closing_snapshot(session: Session, match_id: str,
                      commence_time: datetime) -> OddsSnapshot | None:
    """Последний снимок до старта матча — приближение closing line."""
    return session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match_id)
        .where(OddsSnapshot.captured_at <= commence_time)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def compute_clv_for_anomaly(session: Session, anomaly: Anomaly) -> AnomalyCLV:
    """Посчитать CLV-строку для одной аномалии.

    Всегда возвращает строку (с NULL-полями для не-направленных или при
    отсутствии данных) — это гарантирует, что compute_pending_clv не будет
    пытаться пересчитать ту же аномалию на следующих прогонах.
    """
    bet_side = extract_bet_side(anomaly.detector, anomaly.payload)

    if bet_side is None:
        return AnomalyCLV(
            anomaly_id=anomaly.id, detector=anomaly.detector,
            side=None, prob_at_alert=None, prob_at_close=None, clv_pp=None,
        )

    alert_snap = _alert_snapshot(session, anomaly.match_id, anomaly.detected_at)
    close_snap = _closing_snapshot(session, anomaly.match_id, anomaly.commence_time)

    if alert_snap is None or close_snap is None:
        return AnomalyCLV(
            anomaly_id=anomaly.id, detector=anomaly.detector,
            side=bet_side, prob_at_alert=None, prob_at_close=None, clv_pp=None,
        )

    alert_probs = consensus_probabilities(_reconstruct_match(alert_snap))
    close_probs = consensus_probabilities(_reconstruct_match(close_snap))
    if not alert_probs or not close_probs:
        return AnomalyCLV(
            anomaly_id=anomaly.id, detector=anomaly.detector,
            side=bet_side, prob_at_alert=None, prob_at_close=None, clv_pp=None,
        )

    p_alert = alert_probs.get(bet_side)
    p_close = close_probs.get(bet_side)
    if p_alert is None or p_close is None:
        return AnomalyCLV(
            anomaly_id=anomaly.id, detector=anomaly.detector,
            side=bet_side, prob_at_alert=None, prob_at_close=None, clv_pp=None,
        )

    clv_pp = (p_close - p_alert) * 100.0
    return AnomalyCLV(
        anomaly_id=anomaly.id, detector=anomaly.detector,
        side=bet_side, prob_at_alert=p_alert,
        prob_at_close=p_close, clv_pp=clv_pp,
    )


def compute_pending_clv() -> int:
    """Найти аномалии после старта матча без CLV-строки, посчитать и сохранить.

    Возвращает число записанных строк. Вызывается шедулером ежечасно.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - _POST_KICKOFF_BUFFER
    written = 0

    with SessionLocal() as session:
        existing_subq = select(AnomalyCLV.anomaly_id)
        pending = session.execute(
            select(Anomaly)
            .where(Anomaly.commence_time <= cutoff)
            .where(Anomaly.id.notin_(existing_subq))
        ).scalars().all()

        for a in pending:
            row = compute_clv_for_anomaly(session, a)
            session.add(row)
            written += 1

        if written:
            session.commit()

    if written:
        log.info("CLV: записано %d новых результатов", written)
    return written
