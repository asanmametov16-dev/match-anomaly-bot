"""Тесты precision-gate classify_signal — «точный сигнал» vs «слабое».

Чистая логика. CLV-множители подменяются через calibration._multipliers.
"""
from __future__ import annotations

import pytest

import src.calibration as calib
import src.detectors as det
from src.detectors import AnomalyHit, classify_signal


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
