"""Тесты чистого хелпера notifier._format_gate_line."""
from __future__ import annotations

from src.notifier import _format_gate_line


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
