"""Тесты precision-gate classify_signal — «точный сигнал» vs «слабое».

Чистая логика. CLV-множители подменяются через calibration._multipliers.
"""
from __future__ import annotations

import pytest

import src.calibration as calib
import src.detectors as det
from src.detectors import AnomalyHit, classify_signal, compute_score


@pytest.fixture
def relaxed(monkeypatch):
    """Детерминированные пороги: изолируем проверку направленности."""
    monkeypatch.setattr(det.settings, "signal_gate_enabled", True)
    monkeypatch.setattr(det.settings, "signal_min_detectors", 1)
    monkeypatch.setattr(det.settings, "signal_score_threshold", 0.0)
    monkeypatch.setattr(det.settings, "signal_min_clv_multiplier", 0.0)
    monkeypatch.setattr(det.settings, "signal_min_agreement", 0.5)
    monkeypatch.setattr(det.settings, "signal_min_directional", 2)


@pytest.fixture(autouse=True)
def clean_clv_cache():
    calib._multipliers.clear()
    calib._stats.clear()
    yield
    calib._multipliers.clear()
    calib._stats.clear()


def _hit(detector: str, payload: dict, severity: float = 1.0) -> AnomalyHit:
    return AnomalyHit(detector=detector, severity=severity,
                      description="x", payload=payload)


def _home(detector: str) -> AnomalyHit:
    """Направленный хит, «ставочная сторона» = home (по extract_bet_side)."""
    if detector == "drift":
        return _hit("drift", {"outcome": "home", "drift_pp": 5.0})
    if detector == "synchronized":
        return _hit("synchronized", {"outcome": "home", "direction": "↓"})
    if detector == "sharp_move":
        return _hit("sharp_move", {"outcome": "home"})
    if detector == "model_gap":
        return _hit("model_gap", {"outcome": "home", "market": 2.5, "fair": 2.2})
    raise ValueError(detector)


def test_strong_cluster_is_signal():
    hits = [_home("synchronized"), _home("sharp_move"), _home("drift")]
    label, meta = classify_signal(hits)
    assert label == "signal"
    assert meta["n_detectors"] == 3
    assert meta["side"] == "home"
    assert meta["agreement"] == pytest.approx(1.0)


def test_single_nondirectional_is_weak():
    hits = [_hit("spread", {"outcome": "home", "spread_pp": 9.0})]
    label, meta = classify_signal(hits)
    assert label == "weak"
    assert meta["side"] is None


def test_conflicting_directions_weak():
    hits = [
        _hit("sharp_move", {"outcome": "home"}),
        _hit("drift", {"outcome": "away", "drift_pp": 5.0}),
    ]
    label, meta = classify_signal(hits)
    assert meta["agreement"] == pytest.approx(0.5)
    assert label == "weak"  # < signal_min_agreement (0.55)


def test_clv_downweighted_detectors_blocked(monkeypatch):
    """Счёт ≥ порога, но детекторы исторически CLV-слабые → не сигнал."""
    monkeypatch.setitem(calib._multipliers, "sharp_move", 0.5)
    hits = [_home("synchronized"), _home("sharp_move")]
    # score = 3*1.0 + 2*0.5 = 4.0 ≥ 4, но mean_mult = 0.75 < 1.0
    label, meta = classify_signal(hits)
    assert meta["score"] == pytest.approx(4.0)
    assert meta["mean_clv_mult"] == pytest.approx(0.75)
    assert label == "weak"


def test_gate_disabled_always_signal(monkeypatch):
    monkeypatch.setattr(det.settings, "signal_gate_enabled", False)
    label, meta = classify_signal(
        [_hit("spread", {"outcome": "home", "spread_pp": 9.0})]
    )
    assert label == "signal"
    assert meta["confidence"] == "signal"


def test_empty_cluster_weak_no_crash():
    label, meta = classify_signal([])
    assert label == "weak"
    assert meta["n_detectors"] == 0


# --- доработка: ≥2 РАЗНЫХ направленных детектора + дедуп веса ---------------

def test_single_directional_detector_weak(relaxed):
    """Кейс O'Higgins: один направленный (sharp_move) + ненаправленный
    объём. Всё прочее пройдено, но направление на 1 детекторе → weak."""
    hits = [
        _hit("sharp_move", {"outcome": "home"}),
        _hit("spread", {"outcome": "home", "spread_pp": 9.0}),
        _hit("spread", {"outcome": "away", "spread_pp": 8.0}),
    ]
    label, meta = classify_signal(hits)
    assert meta["n_directional"] == 1
    assert meta["side_detectors"] == 1
    assert meta["agreement"] == pytest.approx(1.0)  # 1/1 — мнимое «100%»
    assert label == "weak"  # side_detectors < signal_min_directional (2)


def test_two_directional_detectors_signal(relaxed):
    hits = [
        _hit("sharp_move", {"outcome": "home"}),
        _hit("drift", {"outcome": "home", "drift_pp": 5.0}),
        _hit("spread", {"outcome": "away", "spread_pp": 8.0}),
    ]
    label, meta = classify_signal(hits)
    assert meta["n_directional"] == 2
    assert meta["side_detectors"] == 2
    assert meta["side"] == "home"
    assert label == "signal"


def test_compute_score_dedupes_multi_hit_detector():
    """spread на home+draw+away = 3 хита, но вес считается ОДИН раз."""
    one = compute_score([_hit("spread", {"outcome": "home"})])
    three = compute_score([
        _hit("spread", {"outcome": "home"}),
        _hit("spread", {"outcome": "draw"}),
        _hit("spread", {"outcome": "away"}),
    ])
    assert three == pytest.approx(one)
