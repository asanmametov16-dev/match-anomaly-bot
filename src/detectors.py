"""Детекторы аномалий в линии коэффициентов.

Каждый детектор — функция, принимающая контекст и возвращающая список
найденных аномалий (или пустой список). Разделение по детекторам сделано
специально: проще тюнить пороги по отдельности и проще отключать.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

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
}

# Детекторы, которые попадают в Telegram-алерт.
# exotic_spread слишком шумный — сохраняется в БД, но не алертится.
ALERT_DETECTORS: frozenset[str] = frozenset({
    "synchronized", "sharp_move", "drift", "spread", "model_gap",
})
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import OddsSnapshot
from .elo import fair_odds_1x2, get_rating
from .odds_client import MatchOdds


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

    hits: list[AnomalyHit] = []
    threshold_pp = settings.spread_pp_threshold

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
                    f"= {spread_pp:.1f}пп (порог {threshold_pp:.1f}пп)"
                ),
                payload={
                    "outcome": outcome,
                    "lo_bm": lo_bm, "lo_prob": lo_p, "lo_odds": lo_odds,
                    "hi_bm": hi_bm, "hi_prob": hi_p, "hi_odds": hi_odds,
                    "spread_pp": spread_pp,
                },
            ))
    return hits


# --- Детектор 2: drift (движение линии) -------------------------------------
def detect_drift(session: Session, match: MatchOdds,
                 current_medians: dict[str, float | None]) -> list[AnomalyHit]:
    """Сравнивает текущую медиану коэффициентов с самым ранним сохранённым
    снимком по этому матчу. Резкое движение — потенциальный сигнал."""
    first = session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match.match_id)
        .order_by(OddsSnapshot.captured_at.asc())
        .limit(1)
    ).scalar_one_or_none()

    if first is None:
        return []  # это первый снимок, сравнивать не с чем

    hits: list[AnomalyHit] = []
    pairs = [
        ("home", first.median_home, current_medians.get("home")),
        ("draw", first.median_draw, current_medians.get("draw")),
        ("away", first.median_away, current_medians.get("away")),
    ]
    for outcome, opening, current in pairs:
        if opening is None or current is None or opening <= 1.0:
            continue
        drift = abs(current - opening) / opening
        if drift >= settings.drift_threshold:
            direction = "↓" if current < opening else "↑"
            hits.append(AnomalyHit(
                detector="drift",
                severity=drift,
                description=(
                    f"Движение по {outcome}: {opening:.2f} {direction} {current:.2f} "
                    f"({drift*100:.1f}%, порог {settings.drift_threshold*100:.0f}%)"
                ),
                payload={"outcome": outcome, "opening": opening,
                         "current": current},
            ))
    return hits


# --- Детектор 3: synchronized — синхронные движения у нескольких букмекеров ---
def detect_synchronized(session: Session, match: MatchOdds) -> list[AnomalyHit]:
    """Сравнивает текущие коэффициенты с предыдущим снимком (а не с открытием)
    и считает, у скольких букмекеров одновременно произошло заметное
    движение в одну сторону.

    Если 3+ букмекера за один интервал двинули коэффициент >5% в одну сторону —
    это сильный сигнал, что 'умные деньги' зашли через несколько контор сразу.
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
            if len(movers) >= min_movers:
                avg_change = sum(
                    abs(c - p) / p for _, p, c in movers
                ) / len(movers)
                hits.append(AnomalyHit(
                    detector="synchronized",
                    severity=avg_change * len(movers),  # величина × массовость
                    description=(
                        f"Синхронное {direction} по {outcome}: "
                        f"{len(movers)} букмекеров, средний сдвиг {avg_change*100:.1f}% "
                        f"(порог {min_movers}+ контор × {threshold*100:.0f}%)"
                    ),
                    payload={"outcome": outcome, "direction": direction,
                             "movers": [{"bm": bm, "from": p, "to": c}
                                        for bm, p, c in movers]},
                ))
    return hits


# --- Детектор 4: расхождение с моделью --------------------------------------
def detect_model_gap(session: Session, match: MatchOdds,
                     current_medians: dict[str, float | None]) -> list[AnomalyHit]:
    """Сравнивает рыночные коэффициенты с 'честными' от Elo-модели.

    Внимание: пока рейтинги не откалибровались, шумит. См. комментарий в elo.py.
    """
    rating_home = get_rating(session, match.home_team)
    rating_away = get_rating(session, match.away_team)
    fair = fair_odds_1x2(rating_home, rating_away)

    hits: list[AnomalyHit] = []
    pairs = [
        ("home", fair.home, current_medians.get("home")),
        ("draw", fair.draw, current_medians.get("draw")),
        ("away", fair.away, current_medians.get("away")),
    ]
    for outcome, fair_price, market in pairs:
        if market is None or market <= 1.0:
            continue
        gap = abs(market - fair_price) / fair_price
        if gap >= settings.model_gap_threshold:
            hits.append(AnomalyHit(
                detector="model_gap",
                severity=gap,
                description=(
                    f"Расхождение с моделью по {outcome}: "
                    f"рынок {market:.2f}, модель {fair_price:.2f} "
                    f"({gap*100:.1f}%, порог {settings.model_gap_threshold*100:.0f}%)"
                ),
                payload={"outcome": outcome, "market": market, "fair": fair_price,
                         "rating_home": rating_home, "rating_away": rating_away},
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

    hits: list[AnomalyHit] = []
    threshold = settings.exotic_spread_threshold
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
                    f"{lo:.2f} ↔ {hi:.2f} ({spread*100:.1f}%, "
                    f"порог {threshold*100:.0f}%)"
                ),
                payload={"market": market_name, "point": point, "name": name,
                         "min": lo, "max": hi, "prices": prices},
            ))
    return hits


def compute_score(hits: list[AnomalyHit]) -> float:
    """Взвешенный счёт подозрительности по сработавшим детекторам."""
    return sum(DETECTOR_WEIGHTS.get(h.detector, 1.0) for h in hits)
