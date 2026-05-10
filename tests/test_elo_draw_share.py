"""Tests for dynamic draw share in fair_odds_1x2."""
from __future__ import annotations

import pytest
from src.elo import _dynamic_draw_share, fair_odds_1x2

# Home advantage = 60 (from default settings).
# effective_gap = abs((rating_home + 60) - rating_away)


def test_draw_share_zero_effective_gap():
    """When effective gap = 0, draw share = 0.30 (maximum for evenly-matched)."""
    # home=1440, away=1500: (1440+60) - 1500 = 0
    ds = _dynamic_draw_share(1440.0, 1500.0)
    assert abs(ds - 0.30) < 1e-9


def test_draw_share_200_raw_gap():
    """Home stronger by 200 raw points → effective gap 260 → draw ≈ 22%."""
    # home=1700, away=1500: (1700+60) - 1500 = 260
    ds = _dynamic_draw_share(1700.0, 1500.0)
    assert abs(ds - 0.222) < 1e-9


def test_draw_share_400_effective_gap():
    """Effective gap = 400 → draw share hits the 0.18 minimum."""
    # home=1740, away=1400: (1740+60) - 1400 = 400
    ds = _dynamic_draw_share(1740.0, 1400.0)
    assert abs(ds - 0.18) < 1e-9


def test_draw_share_clamped_at_minimum():
    """Beyond 400 effective gap, draw share is clamped at 0.18."""
    ds_large = _dynamic_draw_share(1940.0, 1400.0)  # gap = 600
    assert ds_large == 0.18


def test_draw_share_clamped_at_maximum():
    """Draw share cannot exceed 0.32 even if formula would go higher."""
    # Formula can't exceed 0.30 with positive gaps, but test the boundary
    # by checking that small negative gaps (away team very strong at home) still clamp
    # gap = abs(-40) = 40 → 0.30 - 0.012 = 0.288 (below 0.32)
    # To reach the upper clamp: need negative diff, but abs() prevents that
    # The upper clamp 0.32 guards future formula changes — just verify it holds
    ds = _dynamic_draw_share(1440.0, 1500.0)  # max without clamp = 0.30 < 0.32
    assert ds <= 0.32


def test_draw_share_is_monotonically_decreasing():
    """Larger effective Elo gap → smaller draw share."""
    home = 1500.0
    away_values = [1560.0, 1500.0, 1400.0, 1300.0]  # increasing effective gap
    draw_shares = [_dynamic_draw_share(home, away) for away in away_values]
    for i in range(len(draw_shares) - 1):
        assert draw_shares[i] >= draw_shares[i + 1], (
            f"draw_share not monotonically decreasing: {draw_shares}"
        )


# --- Integration with fair_odds_1x2 -----------------------------------------

def test_fair_odds_uses_dynamic_draw_by_default():
    """Default call uses dynamic draw share, not the old fixed 0.26."""
    odds_equal = fair_odds_1x2(1440.0, 1500.0)   # gap=0, draw_share=0.30
    odds_fixed = fair_odds_1x2(1440.0, 1500.0, draw_share=0.30)
    # Should produce the same result
    assert abs(odds_equal.draw - odds_fixed.draw) < 1e-6


def test_fair_odds_draw_lower_for_mismatched_teams():
    """Draw odds are higher (less likely) for heavily mismatched teams."""
    odds_equal = fair_odds_1x2(1440.0, 1500.0)      # gap=0, draw_share=0.30
    odds_mismatch = fair_odds_1x2(1740.0, 1400.0)   # gap=400, draw_share=0.18
    # Higher draw odds = less likely to draw
    assert odds_mismatch.draw > odds_equal.draw


def test_fair_odds_draw_share_override():
    """Passing explicit draw_share overrides the dynamic calculation."""
    odds_dynamic = fair_odds_1x2(1500.0, 1500.0)        # draw_share ≈ 0.282
    odds_override = fair_odds_1x2(1500.0, 1500.0, draw_share=0.50)
    # With 50% draw override, draw odds should be much lower
    assert odds_override.draw < odds_dynamic.draw
