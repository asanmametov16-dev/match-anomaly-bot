"""Тесты метода снятия маржи Шина (src/probability.remove_overround).

Чистая логика, без сети. Покрывает:
- сумма ровно 1.0
- корректность favourite-longshot bias (фаворит ↑, аутсайдер ↓ vs пропорц.)
- сохранение порядка исходов
- 2-исходный рынок (без ничьей)
- вырожденные входы (нет маржи / неположительные) → откат на пропорциональный
- переключатель settings.devig_method
"""
from __future__ import annotations

import pytest

import src.probability as prob
from src.probability import (_devig_proportional, _devig_shin,
                             probabilities_from_match, remove_overround)
from src.odds_client import BookmakerOdds


def _raw(home: float, draw: float | None, away: float) -> dict[str, float]:
    d = {"home": 1.0 / home, "away": 1.0 / away}
    if draw is not None:
        d["draw"] = 1.0 / draw
    return d


# --- базовые свойства Шина ---------------------------------------------------

def test_shin_sums_to_one():
    raw = _raw(1.50, 4.50, 7.00)
    q = _devig_shin(raw, sum(raw.values()))
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-9)


def test_shin_corrects_favourite_longshot_bias():
    """Фаворит должен получить вероятность ВЫШE, аутсайдер НИЖЕ, чем при
    пропорциональном методе — это и есть смысл коррекции."""
    raw = _raw(1.50, 4.50, 7.00)
    total = sum(raw.values())
    shin = _devig_shin(raw, total)
    prop = _devig_proportional(raw, total)

    assert shin["home"] > prop["home"]      # фаворит занижался — поднимаем
    assert shin["away"] < prop["away"]      # аутсайдер завышался — опускаем
    assert sum(prop.values()) == pytest.approx(1.0)


def test_shin_preserves_order():
    # implied: home 0.476 > draw 0.303 > away 0.278 — Shin монотонен
    raw = _raw(2.10, 3.30, 3.60)
    q = _devig_shin(raw, sum(raw.values()))
    assert q["home"] > q["draw"] > q["away"]


def test_shin_two_outcomes():
    raw = {"home": 1.0 / 1.90, "away": 1.0 / 2.10}  # рынок без ничьей
    q = _devig_shin(raw, sum(raw.values()))
    assert set(q) == {"home", "away"}
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-9)
    assert q["home"] > q["away"]


# --- вырожденные входы → откат -----------------------------------------------

def test_shin_no_margin_falls_back():
    """B ≤ 1 (нет маржи / арбитраж): Shin вырожден → пропорциональный."""
    raw = {"home": 0.40, "draw": 0.25, "away": 0.30}  # сумма 0.95 < 1
    total = sum(raw.values())
    assert _devig_shin(raw, total) == _devig_proportional(raw, total)


def test_shin_high_margin_still_sums_to_one():
    raw = _raw(1.40, 4.00, 6.00)  # заметная маржа
    total = sum(raw.values())
    assert total > 1.05
    q = _devig_shin(raw, total)
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < v < 1.0 for v in q.values())


# --- переключатель метода ----------------------------------------------------

def test_remove_overround_default_is_shin(monkeypatch):
    monkeypatch.setattr(prob.settings, "devig_method", "shin")
    raw = _raw(1.50, 4.50, 7.00)
    assert remove_overround(raw) == _devig_shin(raw, sum(raw.values()))


def test_remove_overround_proportional_mode(monkeypatch):
    monkeypatch.setattr(prob.settings, "devig_method", "proportional")
    raw = _raw(1.50, 4.50, 7.00)
    assert remove_overround(raw) == _devig_proportional(raw, sum(raw.values()))


def test_probabilities_from_match_uses_configured_method(monkeypatch):
    bm = BookmakerOdds(bookmaker="t", home=1.50, draw=4.50, away=7.00)
    monkeypatch.setattr(prob.settings, "devig_method", "shin")
    shin = probabilities_from_match(bm)
    monkeypatch.setattr(prob.settings, "devig_method", "proportional")
    prop = probabilities_from_match(bm)
    assert shin["home"] > prop["home"]
    assert sum(shin.values()) == pytest.approx(1.0, abs=1e-9)
