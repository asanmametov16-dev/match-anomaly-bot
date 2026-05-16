"""Детекторы аномалий в линии коэффициентов.

Каждый детектор — функция, принимающая контекст и возвращающая список
найденных аномалий (или пустой список). Разделение по детекторам сделано
специально: проще тюнить пороги по отдельности и проще отключать.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# Букмекеры, которые считаются «острыми» (sharp): они двигают рынок первыми
# и отражают «умные деньги». Если их коэф. заметно ниже soft-контор — сигнал.
SHARP_BOOKMAKERS: frozenset[str] = frozenset({
    "pinnacle", "betfair_ex_eu", "betfair_ex_uk", "betfair",
    "matchbook", "smarkets", "lowvig", "betcris",
})

# Веса детекторов для итогового счёта подозрительности.
# synchronized и sharp_move — самые надёжные сигналы «умных денег».
DETECTOR_WEIGHTS: dict[str, float] = {
    "synchronized": 3.0,
    "sharp_move":   2.0,
    "drift":        2.0,
    "spread":       1.0,
    "model_gap":    1.0,
    "exotic_spread": 1.0,
    "cross_market": 1.0,
}

# Детекторы, которые попадают в Telegram-алерт.
# exotic_spread слишком шумный — сохраняется в БД, но не алертится.
# cross_market включён: он на детерминированном DNB-тождестве (не статистика),
# ждать «созревания» незачем; не-направленный → только сила/количество.
ALERT_DETECTORS: frozenset[str] = frozenset({
    "synchronized", "sharp_move", "drift", "spread", "model_gap",
    "cross_market",
})
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .calibration import detector_multiplier
from .config import settings
from .db import OddsSnapshot
from .elo import fair_odds_1x2, get_rating
from .odds_client import BookmakerOdds, MatchOdds

if TYPE_CHECKING:
    from .sstats_client import XgPrediction


@dataclass
class AnomalyHit:
    detector: str
    severity: float           # числовое значение нарушения
    description: str
    payload: dict


def _median(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and v > 1.0]
    if not clean:
        return None
    return statistics.median(clean)


def _time_bucket_info(match: MatchOdds) -> tuple[float, str]:
    """Return (threshold_multiplier, label) for the match's time-to-kick-off bucket.

    Closer to kick-off → lower multiplier → effectively lower threshold → more alerts.
    Far from kick-off → higher multiplier → effectively higher threshold → less noise.
    Uses match.commence_time timezone so it works with both aware and naive datetimes.
    """
    now = datetime.now(match.commence_time.tzinfo)
    hours = max(0.0, (match.commence_time - now).total_seconds() / 3600)

    buckets = settings.time_buckets_hours        # [6, 24, 72]
    multipliers = settings.time_bucket_multipliers  # [0.7, 0.85, 1.0, 1.3]

    for i, boundary in enumerate(buckets):
        if hours < boundary:
            label = f"<{boundary}ч ×{multipliers[i]:.2f}"
            return multipliers[i], label

    label = f">{buckets[-1]}ч ×{multipliers[-1]:.2f}"
    return multipliers[-1], label


# --- Детектор 1: spread между букмекерами -----------------------------------
def detect_spread(match: MatchOdds) -> list[AnomalyHit]:
    """Большой разброс вероятностей между букмекерами по одному исходу.

    Использует вероятности без маржи (remove_overround), а не сырые коэффициенты:
    это убирает влияние разного размера наценки у разных контор и позволяет
    честно сравнивать «мнения» букмекеров об исходе. Порог в процентных пунктах.
    """
    from .probability import probabilities_from_match

    # Собираем (prob, raw_odds) для каждого исхода по каждому букмекеру
    by_outcome: dict[str, list[tuple[str, float, float | None]]] = {
        "home": [], "draw": [], "away": [],
    }
    for bm in match.bookmakers:
        probs = probabilities_from_match(bm)
        if probs is None:
            continue
        for outcome, p in probs.items():
            by_outcome[outcome].append((bm.bookmaker, p, getattr(bm, outcome)))

    multiplier, bucket_label = _time_bucket_info(match)
    hits: list[AnomalyHit] = []
    threshold_pp = settings.spread_pp_threshold * multiplier

    for outcome, entries in by_outcome.items():
        if len(entries) < 3:
            continue
        probs_only = [p for _, p, _ in entries]
        lo_idx = probs_only.index(min(probs_only))
        hi_idx = probs_only.index(max(probs_only))
        lo_bm, lo_p, lo_odds = entries[lo_idx]
        hi_bm, hi_p, hi_odds = entries[hi_idx]
        spread_pp = (hi_p - lo_p) * 100

        if spread_pp >= threshold_pp:
            hits.append(AnomalyHit(
                detector="spread",
                severity=spread_pp,
                description=(
                    f"Расхождение по {outcome}: "
                    f"{lo_p*100:.1f}% ({lo_odds:.2f}) ↔ {hi_p*100:.1f}% ({hi_odds:.2f}) "
                    f"= {spread_pp:.1f}пп [корзина {bucket_label}, эфф. {threshold_pp:.1f}пп]"
                ),
                payload={
                    "outcome": outcome,
                    "lo_bm": lo_bm, "lo_prob": lo_p, "lo_odds": lo_odds,
                    "hi_bm": hi_bm, "hi_prob": hi_p, "hi_odds": hi_odds,
                    "spread_pp": spread_pp, "bucket_label": bucket_label,
                },
            ))
    return hits


# --- Детектор 2: drift (движение линии) -------------------------------------
def detect_drift(session: Session, match: MatchOdds) -> list[AnomalyHit]:
    """Сравнивает текущие вероятности с самым старым снимком в скользящем окне.

    Используем consensus_probabilities(match) для текущего состояния, а не
    последний снимок из БД: detect_drift вызывается ДО сохранения нового снимка,
    поэтому «текущий» снимок ещё не существует. Вычисление прямо из match.bookmakers
    гарантирует, что current всегда актуален и не зависит от порядка записи в БД.
    """
    from .probability import consensus_probabilities

    window_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=settings.drift_window_minutes)

    opening_snap = session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match.match_id)
        .where(OddsSnapshot.captured_at >= window_start)
        .order_by(OddsSnapshot.captured_at.asc())
        .limit(1)
    ).scalar_one_or_none()

    if opening_snap is None:
        return []  # нет снимка в окне — рано сравнивать

    # Восстанавливаем объекты BookmakerOdds из JSON-снимка для честного
    # сравнения через consensus_probabilities (с убранной маржой)
    bm_objects = [
        BookmakerOdds(
            bookmaker=b.get("bookmaker", ""),
            home=b.get("home"),
            draw=b.get("draw"),
            away=b.get("away"),
        )
        for b in (opening_snap.bookmakers or [])
    ]
    opening_match = MatchOdds(
        match_id=match.match_id,
        sport_key=match.sport_key,
        home_team=match.home_team,
        away_team=match.away_team,
        commence_time=match.commence_time,
        bookmakers=bm_objects,
    )
    opening_probs = consensus_probabilities(opening_match)
    if opening_probs is None:
        return []

    current_probs = consensus_probabilities(match)
    if current_probs is None:
        return []

    snap_age_min = (datetime.now(timezone.utc).replace(tzinfo=None) - opening_snap.captured_at).total_seconds() / 60
    multiplier, bucket_label = _time_bucket_info(match)

    hits: list[AnomalyHit] = []
    threshold_pp = settings.drift_pp_threshold * multiplier

    for outcome in ("home", "draw", "away"):
        opening = opening_probs.get(outcome)
        current = current_probs.get(outcome)
        if opening is None or current is None:
            continue
        drift_pp = (current - opening) * 100
        abs_drift_pp = abs(drift_pp)
        if abs_drift_pp >= threshold_pp:
            direction = "↑" if drift_pp > 0 else "↓"
            hits.append(AnomalyHit(
                detector="drift",
                severity=abs_drift_pp,
                description=(
                    f"Движение по {outcome}: "
                    f"{opening*100:.1f}% {direction} {current*100:.1f}% "
                    f"= {abs_drift_pp:.1f}пп за {snap_age_min:.0f}мин "
                    f"[корзина {bucket_label}, эфф. {threshold_pp:.1f}пп]"
                ),
                payload={
                    "outcome": outcome,
                    "opening_prob": opening,
                    "current_prob": current,
                    "drift_pp": drift_pp,
                    "window_minutes": settings.drift_window_minutes,
                    "snap_age_minutes": snap_age_min,
                    "bucket_label": bucket_label,
                },
            ))
    return hits


# --- Детектор 3: synchronized — синхронные движения у нескольких букмекеров ---
def detect_synchronized(session: Session, match: MatchOdds) -> list[AnomalyHit]:
    """Сравнивает текущие коэффициенты с предыдущим снимком (а не с открытием)
    и считает, у скольких букмекеров одновременно произошло заметное
    движение в одну сторону.

    Срабатывает только если среди двинувших есть хотя бы один sharp-букмекер.
    Движение одних лишь soft-контор — это копирование чужой линии, а не сигнал
    «умных денег». Слабые срабатывания (без sharp) логируются на DEBUG.
    """
    prev = session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match.match_id)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if prev is None or not prev.bookmakers:
        return []

    prev_by_bm = {b["bookmaker"]: b for b in prev.bookmakers}
    threshold = settings.sync_move_threshold
    min_movers = settings.sync_min_bookmakers
    _sharp_set = {b.lower() for b in settings.sharp_bookmakers}
    hits: list[AnomalyHit] = []

    for outcome in ("home", "draw", "away"):
        movers_down: list[tuple[str, float, float]] = []
        movers_up: list[tuple[str, float, float]] = []
        for current in match.bookmakers:
            previous = prev_by_bm.get(current.bookmaker)
            if previous is None:
                continue
            cur_price = getattr(current, outcome)
            prev_price = previous.get(outcome)
            if cur_price is None or prev_price is None or prev_price <= 1.0:
                continue
            change = (cur_price - prev_price) / prev_price
            if change <= -threshold:
                movers_down.append((current.bookmaker, prev_price, cur_price))
            elif change >= threshold:
                movers_up.append((current.bookmaker, prev_price, cur_price))

        for direction, movers in (("↓", movers_down), ("↑", movers_up)):
            if len(movers) < min_movers:
                continue
            sharp_movers = [bm for bm, _, _ in movers if bm.lower() in _sharp_set]
            if not sharp_movers:
                log.debug(
                    "Слабое синхр. %s по %s: %d контор без sharp — пропускаем %s",
                    direction, outcome, len(movers),
                    [bm for bm, _, _ in movers],
                )
                continue
            avg_change = sum(abs(c - p) / p for _, p, c in movers) / len(movers)
            hits.append(AnomalyHit(
                detector="synchronized",
                severity=avg_change * len(movers),
                description=(
                    f"Синхронное {direction} по {outcome}: "
                    f"{len(movers)} букмекеров (sharp: {', '.join(sharp_movers)}), "
                    f"средний сдвиг {avg_change*100:.1f}% "
                    f"(порог {min_movers}+ × {threshold*100:.0f}%)"
                ),
                payload={"outcome": outcome, "direction": direction,
                         "sharp_movers": sharp_movers,
                         "movers": [{"bm": bm, "from": p, "to": c}
                                    for bm, p, c in movers]},
            ))
    return hits


# --- Детектор 4: расхождение с моделью --------------------------------------
def detect_model_gap(session: Session, match: MatchOdds,
                     current_medians: dict[str, float | None],
                     xg_pred: "XgPrediction | None" = None) -> list[AnomalyHit]:
    """Сравнивает рыночные коэффициенты с 'честными' от модели.

    Приоритет референс-модели (точность убывает):
      1. xg_pred (sstats.net winProb)          → source "sstats_xg"
      2. маржа-free консенсус sharp-контор      → source "sharp_consensus"
      3. Elo (холодный старт, шумит 1-2 недели) → source "elo"
    Источник сохраняется в payload["source"].
    """
    from .probability import sharp_consensus_probabilities
    from .sstats_history import league_model_trust

    if xg_pred is not None:
        fair_home = 1.0 / max(xg_pred.home_win_prob, 0.01)
        fair_draw = 1.0 / max(xg_pred.draw_prob, 0.01)
        fair_away = 1.0 / max(xg_pred.away_win_prob, 0.01)
        source = "sstats_xg"
        model_info: dict = {
            "home_xg": xg_pred.home_xg,
            "away_xg": xg_pred.away_xg,
            "home_glicko": xg_pred.home_glicko,
            "away_glicko": xg_pred.away_glicko,
            "league": xg_pred.league,
            "model_trust": league_model_trust(xg_pred.league),
        }
    elif (sharp := sharp_consensus_probabilities(
            match, settings.model_gap_min_sharp_books)) is not None:
        fair_home = 1.0 / max(sharp["home"], 0.01)
        fair_draw = 1.0 / max(sharp["draw"], 0.01)
        fair_away = 1.0 / max(sharp["away"], 0.01)
        source = "sharp_consensus"
        model_info = {
            "sharp_p_home": sharp["home"],
            "sharp_p_draw": sharp["draw"],
            "sharp_p_away": sharp["away"],
        }
    else:
        rating_home = get_rating(session, match.home_team)
        rating_away = get_rating(session, match.away_team)
        fair = fair_odds_1x2(rating_home, rating_away)
        fair_home, fair_draw, fair_away = fair.home, fair.draw, fair.away
        source = "elo"
        model_info = {"rating_home": rating_home, "rating_away": rating_away}

    multiplier, bucket_label = _time_bucket_info(match)
    threshold = settings.model_gap_threshold * multiplier

    hits: list[AnomalyHit] = []
    pairs = [
        ("home", fair_home, current_medians.get("home")),
        ("draw", fair_draw, current_medians.get("draw")),
        ("away", fair_away, current_medians.get("away")),
    ]
    for outcome, fair_price, market in pairs:
        if market is None or market <= 1.0:
            continue
        gap = abs(market - fair_price) / fair_price
        if gap >= threshold:
            hits.append(AnomalyHit(
                detector="model_gap",
                severity=gap,
                description=(
                    f"Расхождение с моделью ({source}) по {outcome}: "
                    f"рынок {market:.2f}, модель {fair_price:.2f} "
                    f"({gap*100:.1f}%) [корзина {bucket_label}, эфф. {threshold*100:.0f}%]"
                ),
                payload={"outcome": outcome, "market": market, "fair": fair_price,
                         "source": source, "bucket_label": bucket_label,
                         **model_info},
            ))
    return hits


# --- Детектор 5: sharp_move — sharp vs soft букмекеры ----------------------

def _margin_normalized_prob(bm: "BookmakerOdds", outcome: str) -> float | None:  # noqa: F821
    """Вероятность исхода с нормализацией по марже конкретного букмекера.

    Сырые коэффициенты нельзя сравнивать напрямую: биржи (Betfair, Smarkets)
    не имеют маржи и дают ВСЕГДА более высокие коэффициенты, чем обычные конторы.
    После нормализации 1/odds / sum(1/odds) маржа уходит и можно честно сравнивать
    «кого букмекер считает фаворитом».
    """
    prices = {"home": bm.home, "draw": bm.draw, "away": bm.away}
    clean = {o: p for o, p in prices.items() if p is not None and p > 1.0}
    if outcome not in clean or len(clean) < 2:
        return None
    raw = {o: 1.0 / p for o, p in clean.items()}
    total = sum(raw.values())
    return raw[outcome] / total


def detect_sharp_move(match: MatchOdds) -> list[AnomalyHit]:
    """Сравнивает нормализованные вероятности sharp-контор против soft.

    Сравниваем не сырые коэффициенты, а вероятности внутри линии каждого
    букмекера (с поправкой на маржу). Если Pinnacle/Betfair считают исход X
    значительно вероятнее, чем soft-книги — это сигнал «умных денег».
    """
    hits: list[AnomalyHit] = []
    threshold = settings.sharp_move_threshold  # минимальная разница в вероятности (3pp по умолч.)

    for outcome in ("home", "draw", "away"):
        sharp_probs: list[tuple[str, float]] = []
        soft_probs: list[tuple[str, float]] = []

        for bm in match.bookmakers:
            prob = _margin_normalized_prob(bm, outcome)
            if prob is None:
                continue
            if bm.bookmaker.lower() in SHARP_BOOKMAKERS:
                sharp_probs.append((bm.bookmaker, prob))
            else:
                soft_probs.append((bm.bookmaker, prob))

        if not sharp_probs or len(soft_probs) < 2:
            continue

        sharp_med = statistics.median([p for _, p in sharp_probs])
        soft_med = statistics.median([p for _, p in soft_probs])

        # Sharp выше soft → sharps backing этот исход сильнее
        diff = sharp_med - soft_med
        if diff >= threshold:
            sharp_names = ", ".join(b for b, _ in sharp_probs)
            hits.append(AnomalyHit(
                detector="sharp_move",
                severity=diff,
                description=(
                    f"Sharp-движение по {outcome}: "
                    f"sharp {sharp_med*100:.1f}% vs soft {soft_med*100:.1f}% "
                    f"(+{diff*100:.1f}пп, порог {threshold*100:.0f}пп)"
                    f" [{sharp_names}]"
                ),
                payload={
                    "outcome": outcome,
                    "sharp_prob": sharp_med,
                    "soft_prob": soft_med,
                    "diff": diff,
                    "sharp_books": [b for b, _ in sharp_probs],
                    "soft_books": [b for b, _ in soft_probs],
                },
            ))
    return hits


# --- Детектор 6: exotic_spread — расхождение на тоталах и форах -------------
def detect_exotic_spread(match: MatchOdds) -> list[AnomalyHit]:
    """Большое расхождение между букмекерами на тоталах/форах.

    Этим рынкам уделяется меньше внимания, чем 1X2, и аномалии тут
    встречаются чаще. Группируем по (market_type, point, name): сравниваем
    Over 2.5 одного букмекера с Over 2.5 другого, не Over 2.5 с Over 3.5.
    """
    # Собираем: {(market, point, name): [(bookmaker, price), ...]}
    grouped: dict[tuple[str, float, str], list[tuple[str, float]]] = {}

    for bm in match.bookmakers:
        for market_name, items in (("totals", bm.totals), ("spreads", bm.spreads)):
            if not items:
                continue
            for item in items:
                price = item.get("price")
                point = item.get("point")
                name = item.get("name")
                if price is None or point is None or name is None or price <= 1.0:
                    continue
                grouped.setdefault((market_name, point, name), []).append(
                    (bm.bookmaker, price)
                )

    multiplier, bucket_label = _time_bucket_info(match)
    hits: list[AnomalyHit] = []
    threshold = settings.exotic_spread_threshold * multiplier
    for (market_name, point, name), prices in grouped.items():
        if len(prices) < 3:
            continue
        values = [p for _, p in prices]
        lo, hi = min(values), max(values)
        spread = (hi - lo) / lo
        if spread >= threshold:
            hits.append(AnomalyHit(
                detector="exotic_spread",
                severity=spread,
                description=(
                    f"Расхождение на {market_name} {name} {point}: "
                    f"{lo:.2f} ↔ {hi:.2f} ({spread*100:.1f}%) "
                    f"[корзина {bucket_label}, эфф. {threshold*100:.0f}%]"
                ),
                payload={"market": market_name, "point": point, "name": name,
                         "min": lo, "max": hi, "prices": prices,
                         "bucket_label": bucket_label},
            ))
    return hits


# --- Детектор 7: cross_market — h2h vs азиатская фора 0.0 ------------------
def detect_cross_market(match: MatchOdds) -> list[AnomalyHit]:
    """Межрыночная несогласованность: 1X2 против азиатской форы на линии 0.0.

    Фора 0.0 (level ball) = Draw-No-Bet: при ничьей ставка возвращается.
    Значит её маржа-free вероятность по home обязана совпадать с DNB,
    выведенной из 1X2: p_home / (p_home + p_away). Это **тождество**, а не
    модель — расхождение указывает на устаревшую линию/ошибку в одном из
    рынков. Ортогонально одиночным детекторам и почти без ложных
    срабатываний; не-направленный (как spread/exotic).

    No-op, если форы 0.0 нет хотя бы у cross_market_min_books контор или
    не считается консенсус 1X2 — лучше молчать, чем шуметь.
    """
    from .probability import consensus_probabilities, remove_overround

    h2h = consensus_probabilities(match)
    if not h2h or h2h.get("home") is None or h2h.get("away") is None:
        return []
    denom = h2h["home"] + h2h["away"]
    if denom <= 0:
        return []
    dnb_h2h = h2h["home"] / denom

    dnb_ah: list[float] = []
    for bm in match.bookmakers:
        if not bm.spreads:
            continue
        price_h = price_a = None
        for item in bm.spreads:
            point, name, price = item.get("point"), item.get("name"), item.get("price")
            if price is None or price <= 1.0 or point is None:
                continue
            try:
                if abs(float(point)) > 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
            if name == match.home_team:
                price_h = price
            elif name == match.away_team:
                price_a = price
        if price_h and price_a:
            tw = remove_overround({"home": 1.0 / price_h, "away": 1.0 / price_a})
            dnb_ah.append(tw["home"])

    if len(dnb_ah) < settings.cross_market_min_books:
        return []

    ah_consensus = statistics.median(dnb_ah)
    gap_pp = abs(dnb_h2h - ah_consensus) * 100.0

    multiplier, bucket_label = _time_bucket_info(match)
    threshold_pp = settings.cross_market_pp_threshold * multiplier
    if gap_pp < threshold_pp:
        return []

    return [AnomalyHit(
        detector="cross_market",
        severity=gap_pp,
        description=(
            f"Межрыночное расхождение DNB: 1X2 даёт {dnb_h2h*100:.1f}%, "
            f"фора 0.0 — {ah_consensus*100:.1f}% ({gap_pp:.1f}пп, "
            f"{len(dnb_ah)} контор) [корзина {bucket_label}, "
            f"эфф. {threshold_pp:.1f}пп]"
        ),
        payload={
            "dnb_h2h": dnb_h2h,
            "dnb_ah0": ah_consensus,
            "gap_pp": gap_pp,
            "ah_books": len(dnb_ah),
            "bucket_label": bucket_label,
        },
    )]


def compute_score(hits: list[AnomalyHit]) -> float:
    """Взвешенный счёт подозрительности по сработавшим детекторам.

    Базовый вес (DETECTOR_WEIGHTS) масштабируется CLV-множителем: детектор,
    чьи срабатывания исторически не подтверждались движением рынка, весит
    меньше. См. calibration.py.
    """
    return sum(
        DETECTOR_WEIGHTS.get(h.detector, 1.0) * detector_multiplier(h.detector)
        for h in hits
    )


def classify_signal(hits: list[AnomalyHit]) -> tuple[str, dict]:
    """Precision-gate: «точный сигнал» против «слабого наблюдения».

    Точный сигнал = несколько разных детекторов + высокий CLV-взвешенный
    счёт + детекторы исторически CLV-подтверждены (множитель ≥ порога) +
    однонаправленный консенсус. Слабые кластеры всё равно сохраняются
    (помечаются confidence=weak), но не эскалируются в Telegram.

    Это НЕ ставочная рекомендация — лишь оценка качества рыночного сигнала.
    Возвращает (label ∈ {"signal","weak"}, meta).
    """
    from collections import Counter

    from .clv import extract_bet_side

    # model_gap из лиги с ненадёжной sstats-моделью не участвует в решении
    # о «сигнале» (но сохраняется и помечается отдельно). См. #3.
    dropped_untrusted = sum(
        1 for h in hits
        if h.detector == "model_gap"
        and (h.payload or {}).get("model_trust") == "unreliable"
    )
    hits = [
        h for h in hits
        if not (h.detector == "model_gap"
                and (h.payload or {}).get("model_trust") == "unreliable")
    ]

    distinct = sorted({h.detector for h in hits})
    n = len(distinct)
    score = compute_score(hits)
    mults = [detector_multiplier(d) for d in distinct] or [1.0]
    mean_mult = sum(mults) / len(mults)

    sides = [s for h in hits
             if (s := extract_bet_side(h.detector, h.payload or {}))]
    agreement, agreed_side = 0.0, None
    if sides:
        agreed_side, top = Counter(sides).most_common(1)[0]
        agreement = top / len(sides)

    meta = {
        "n_detectors": n,
        "score": round(score, 2),
        "mean_clv_mult": round(mean_mult, 2),
        "agreement": round(agreement, 2),
        "side": agreed_side,
        "dropped_untrusted_model_gap": dropped_untrusted,
    }

    if not settings.signal_gate_enabled:
        meta["confidence"] = "signal"
        return "signal", meta

    is_signal = (
        n >= settings.signal_min_detectors
        and score >= settings.signal_score_threshold
        and mean_mult >= settings.signal_min_clv_multiplier
        and bool(sides)  # нужен actionable направленный консенсус
        and agreement >= settings.signal_min_agreement
    )
    label = "signal" if is_signal else "weak"
    meta["confidence"] = label
    return label, meta
