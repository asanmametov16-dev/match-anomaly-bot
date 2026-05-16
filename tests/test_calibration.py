"""Тесты CLV-калибровки весов детекторов.

In-memory SQLite, без сети. Покрывает:
- маппинг среднего CLV → множитель (clamp, монотонность, центр в 1.0)
- порог min_samples (мало данных → детектор не калибруется)
- refresh_detector_weights end-to-end по таблице AnomalyCLV
- учёт флага clv_calibration_enabled
- интеграция с compute_score
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.calibration as calib
from src.calibration import (current_calibration, detector_multiplier,
                             multiplier_for_mean_clv, refresh_detector_weights)
from src.db import AnomalyCLV, Base
from src.detectors import DETECTOR_WEIGHTS, AnomalyHit, compute_score


@pytest.fixture
def db(monkeypatch):
    """In-memory engine + SessionLocal калибровки указывает на него."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(calib, "SessionLocal", Session)
    yield Session


@pytest.fixture(autouse=True)
def clean_cache():
    calib._multipliers.clear()
    calib._stats.clear()
    yield
    calib._multipliers.clear()
    calib._stats.clear()


def _add_clv(session, detector: str, n: int, clv_pp: float, start_id: int):
    for i in range(n):
        session.add(AnomalyCLV(
            anomaly_id=start_id + i, detector=detector, side="home",
            prob_at_alert=0.5, prob_at_close=0.5 + clv_pp / 100.0,
            clv_pp=clv_pp,
        ))


# --- multiplier_for_mean_clv -------------------------------------------------

def test_zero_clv_is_neutral():
    assert multiplier_for_mean_clv(0.0) == pytest.approx(1.0)


def test_positive_clv_boosts_monotonic():
    m1 = multiplier_for_mean_clv(1.0)
    m2 = multiplier_for_mean_clv(2.0)
    assert 1.0 < m1 < m2


def test_negative_clv_damps():
    assert multiplier_for_mean_clv(-2.0) < 1.0


def test_clamped_to_config_bounds(monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_min_multiplier", 0.3)
    monkeypatch.setattr(calib.settings, "clv_calibration_max_multiplier", 1.6)
    assert multiplier_for_mean_clv(1000.0) == pytest.approx(1.6)
    assert multiplier_for_mean_clv(-1000.0) == pytest.approx(0.3)


# --- refresh_detector_weights ------------------------------------------------

def test_refresh_calibrates_only_above_min_samples(db, monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_min_samples", 30)
    with db() as s:
        _add_clv(s, "drift", n=40, clv_pp=2.0, start_id=1)        # хватает данных
        _add_clv(s, "model_gap", n=5, clv_pp=-3.0, start_id=1000)  # мало
        s.commit()

    refresh_detector_weights()

    assert detector_multiplier("drift") > 1.0          # +CLV → буст
    assert detector_multiplier("model_gap") == 1.0     # мало данных → нейтрально


def test_refresh_negative_clv_downweights(db, monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_min_samples", 10)
    with db() as s:
        _add_clv(s, "spread", n=20, clv_pp=-4.0, start_id=1)
        s.commit()

    refresh_detector_weights()
    assert detector_multiplier("spread") < 1.0


def test_refresh_ignores_null_clv_rows(db, monkeypatch):
    """Sentinel-строки (clv_pp NULL) не должны влиять на среднее/счётчик."""
    monkeypatch.setattr(calib.settings, "clv_calibration_min_samples", 3)
    with db() as s:
        _add_clv(s, "drift", n=4, clv_pp=1.0, start_id=1)
        for i in range(50):  # 50 NULL — не должны считаться
            s.add(AnomalyCLV(anomaly_id=5000 + i, detector="drift",
                              side=None, clv_pp=None))
        s.commit()

    refresh_detector_weights()
    n, mean_clv, _ = current_calibration()["drift"]
    assert n == 4
    assert mean_clv == pytest.approx(1.0)


# --- detector_multiplier флаг ------------------------------------------------

def test_disabled_flag_forces_neutral(db, monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_min_samples", 5)
    with db() as s:
        _add_clv(s, "drift", n=20, clv_pp=5.0, start_id=1)
        s.commit()
    refresh_detector_weights()

    monkeypatch.setattr(calib.settings, "clv_calibration_enabled", False)
    assert detector_multiplier("drift") == 1.0


# --- интеграция с compute_score ---------------------------------------------

def test_compute_score_applies_multiplier(monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_enabled", True)
    # drift base weight = 2.0; зануляем эффект множителем 0.5 → 1.0
    calib._multipliers["drift"] = 0.5
    hit = AnomalyHit(detector="drift", severity=1.0, description="x", payload={})

    base = DETECTOR_WEIGHTS["drift"]
    assert compute_score([hit]) == pytest.approx(base * 0.5)


def test_compute_score_uncalibrated_uses_base(monkeypatch):
    monkeypatch.setattr(calib.settings, "clv_calibration_enabled", True)
    # пустой кэш → множитель 1.0 → чистый базовый вес
    hit = AnomalyHit(detector="sharp_move", severity=1.0, description="x", payload={})
    assert compute_score([hit]) == pytest.approx(DETECTOR_WEIGHTS["sharp_move"])
