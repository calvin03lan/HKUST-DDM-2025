"""
Tier-0 and optional degradation-aware What-If gap projections (Part D).

Pure functions — pass numbers from teammate B lap pipeline or placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass


def pace_delta_seconds_per_lap(
    opp_lap_seconds: float, pia_lap_seconds: float
) -> float:
    """Return T_OPP − T_PIA; positive ⇒ Piastri faster on average."""
    return opp_lap_seconds - pia_lap_seconds


def gap_final_simple(*, gap0_s: float, delta_p_s_per_lap: float, laps_remaining: int) -> float:
    """
    G_final ≈ G0 + ΔP · N — see docs/what_if.md Tier 0.

    All times in seconds; laps_remaining counted after the modelling anchor point.
    """
    if laps_remaining < 0:
        raise ValueError("laps_remaining must be ≥ 0")
    return gap0_s + delta_p_s_per_lap * float(laps_remaining)


@dataclass(frozen=True, slots=True)
class DegradationSlopes:
    """Linear seconds-added per intra-stint lap index (1..N)."""

    d_pia: float = 0.0
    d_opp: float = 0.0


def gap_final_with_linear_deg(
    *,
    gap0_s: float,
    base_opp_lap_s: float,
    base_pia_lap_s: float,
    laps_remaining: int,
    deg: DegradationSlopes,
) -> float:
    """
    Sum_k [ (T_OPP + d_OPP·k) − (T_PIA + d_PIA·k) ] for k = 1..N.
    """
    if laps_remaining < 0:
        raise ValueError("laps_remaining must be ≥ 0")
    total = gap0_s
    for k in range(1, laps_remaining + 1):
        t_opp = base_opp_lap_s + deg.d_opp * k
        t_pia = base_pia_lap_s + deg.d_pia * k
        total += t_opp - t_pia
    return total


def _self_check() -> None:
    # Toy numbers: opponent 92.0 s, Piastri 91.8 s ⇒ ΔP = +0.2 s/lap
    dp = pace_delta_seconds_per_lap(92.0, 91.8)
    assert abs(dp - 0.2) < 1e-9

    g = gap_final_simple(gap0_s=3.5, delta_p_s_per_lap=dp, laps_remaining=25)
    assert abs(g - 8.5) < 1e-9  # 3.5 + 0.2*25 = 8.5

    # Degradation: equal slopes ⇒ same as tier-0 additive total if bases match DP per lap...
    gd = gap_final_with_linear_deg(
        gap0_s=3.5,
        base_opp_lap_s=92.0,
        base_pia_lap_s=91.8,
        laps_remaining=2,
        deg=DegradationSlopes(d_pia=0.01, d_opp=0.03),
    )
    assert gd > gap_final_simple(gap0_s=3.5, delta_p_s_per_lap=dp, laps_remaining=2)

    print("what_if_model self-check: OK")


if __name__ == "__main__":
    _self_check()
