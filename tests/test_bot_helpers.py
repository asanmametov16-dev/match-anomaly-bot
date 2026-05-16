"""Тесты чистых хелперов bot.py (без Telegram-моков)."""
from __future__ import annotations

from src.bot import _conf_label


def test_signal():
    assert _conf_label({"signal_confidence": "signal"}) == ("🎯", "signal")


def test_weak():
    assert _conf_label({"signal_confidence": "weak"}) == ("💤", "weak")


def test_missing_and_none():
    assert _conf_label({}) == ("·", "—")
    assert _conf_label(None) == ("·", "—")
    assert _conf_label({"other": 1}) == ("·", "—")
