"""Тесты notifier._format_gate_line и _format_signal (консолидация)."""
from __future__ import annotations

from datetime import datetime, timezone

from src.notifier import _format_gate_line, _format_signal
from src.detectors import AnomalyHit
from src.odds_client import MatchOdds


def _match():
    return MatchOdds(match_id="m", sport_key="s", home_team="Home FC",
                     away_team="Away FC",
                     commence_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
                     bookmakers=[])


def test_empty_meta_returns_blank():
    assert _format_gate_line(None) == ""
    assert _format_gate_line({}) == ""


def test_renders_side_agreement_detectors():
    line = _format_gate_line({"side": "home", "agreement": 1.0,
                              "n_detectors": 3})
    assert "П1" in line and "100%" in line and "детекторов 3" in line
    assert "model_gap" not in line  # ничего не отсеяно


def test_notes_dropped_untrusted_model_gap():
    line = _format_gate_line({"side": "away", "agreement": 0.67,
                              "n_detectors": 2,
                              "dropped_untrusted_model_gap": 1})
    assert "П2" in line and "67%" in line
    assert "отсеян model_gap×1" in line


def test_unknown_side_falls_back():
    line = _format_gate_line({"side": None, "agreement": 0.5,
                              "n_detectors": 2})
    assert "—" in line


# --- _format_signal: согласованность с gate (баг-репорт) --------------------

def _h(det, payload, sev=1.0):
    return AnomalyHit(detector=det, severity=sev, description="x",
                      payload=payload)


def test_signal_uses_meta_side_no_contradiction():
    """Баг: gate сказал home/100%, severity-эвристика дала бы 'противоречивый'.
    С meta должен взять сторону гейта и НЕ писать 'противоречивый'."""
    hits = [
        _h("sharp_move", {"outcome": "home"}, sev=1.0),
        _h("drift", {"outcome": "away", "drift_pp": 5.0}, sev=1.0),
    ]
    out = _format_signal(hits, _match(), meta={"side": "home"})
    assert "противоречивый" not in out
    assert "Победа хозяев" in out


def test_signal_contradiction_only_without_meta():
    hits = [
        _h("sharp_move", {"outcome": "home"}, sev=1.0),
        _h("drift", {"outcome": "away", "drift_pp": 5.0}, sev=1.0),
    ]
    out = _format_signal(hits, _match(), meta=None)
    assert "противоречивый" in out


def test_signal_empty_when_no_directional_and_no_meta():
    hits = [_h("spread", {"outcome": "home", "spread_pp": 9.0})]
    assert _format_signal(hits, _match(), meta=None) == ""
