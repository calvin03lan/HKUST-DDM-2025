#!/usr/bin/env python3
"""
DRAFT_2026-04-27 — dual-scope pace summary + Tier-0 What-If sensitivity.

Run: uv run python scripts/run_draft_insight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_D = Path(__file__).resolve().parent
sys.path.insert(0, str(_D))

from ingest_team_b import load_csv_path, summarize_pace_delta
from what_if_model import gap_final_simple

DRAFT_TAG = "DRAFT_2026-04-27"

# flow.md scripted Oscar pit Lap 18 -> exploratory laps 19..53 (see sync_with_team_B.md)
SCOPE_A_PIA = (None, None)
SCOPE_A_OPP = (None, None)
SCOPE_B = (19, 53)
SCOPE_C_ALT_MEDIUM = (11, 17)

# Stage D placeholders until race-trace ingestion (seconds; positive ⇒ Piastri ahead)
G0_CANDIDATES_S = [1.6, 3.2, 4.0]
N_CANDIDATES = [15, 25, 35]


def _print_scope(name: str, pia_rng: tuple, opp_rng: tuple, path: Path) -> dict[str, float]:
    stats = summarize_pace_delta(
        path, stint_pia=pia_rng, stint_opp=opp_rng
    )
    print(f"\n=== {name} | {DRAFT_TAG} ===")
    print(f"  mean_pia_s:        {stats['mean_pia_s']:.4f}")
    print(f"  mean_opp_s (ANT):  {stats['mean_opp_s']:.4f}")
    print(f"  delta_p_s_per_lap: {stats['delta_p_s_per_lap']:+.4f}  (>0 ⇒ Piastri faster per lap)")
    return stats


def _sensitivity_table(delta_p: float, label: str) -> None:
    print(f"\n--- Tier-0 What-If grid ({label}) ΔP={delta_p:+.4f} s/lap ---")
    print(f"{'G0_s':>8} | " + " | ".join(f"N={n:>2}" for n in N_CANDIDATES))
    for g0 in G0_CANDIDATES_S:
        row = [f"{gap_final_simple(gap0_s=g0, delta_p_s_per_lap=delta_p, laps_remaining=n):+7.2f}" for n in N_CANDIDATES]
        print(f"{g0:8.2f} | " + " | ".join(row))
    print("  (Units: projected finish gap in seconds; positive ⇒ Piastri ahead under Tier-0 assumptions.)")


def main() -> None:
    path = load_csv_path()
    print(f"CSV source: {path}")
    print(f"Tag: {DRAFT_TAG} — exploratory only until teammate B freeze.")

    a_stats = _print_scope("Scope_A_full_race", SCOPE_A_PIA, SCOPE_A_OPP, path)
    b_stats = _print_scope(
        "Scope_B_post_scripted_pit_L19-L53_PIA_ANT",
        SCOPE_B,
        SCOPE_B,
        path,
    )
    _print_scope(
        "Scope_C_alt_medium_tyres_only_L11-L17_exploratory",
        SCOPE_C_ALT_MEDIUM,
        SCOPE_C_ALT_MEDIUM,
        path,
    )

    _sensitivity_table(a_stats["delta_p_s_per_lap"], "full_race_Scope_A")
    _sensitivity_table(b_stats["delta_p_s_per_lap"], "stint_Window_Scope_B")

    print("\n--- ±10% on |ΔP| (Scope_B) ---")
    dp = b_stats["delta_p_s_per_lap"]
    for mult in (0.9, 1.0, 1.1):
        dpm = dp * mult
        g = gap_final_simple(gap0_s=3.2, delta_p_s_per_lap=dpm, laps_remaining=25)
        print(f"  ΔP×{mult:.1f}: {dpm:+.4f} s/lap  →  G_final @ G0=3.2, N=25: {g:+.3f} s")


if __name__ == "__main__":
    main()
