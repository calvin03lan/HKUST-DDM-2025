# Single source of truth: teammate B lap pipeline vs Part D drafts

**Document version:** `DRAFT_2026-04-27` — numbers below are exploratory until teammate B freezes inputs.

This repository may contain **two independent race-lap snapshots**:

| Source | Path / mechanism | Role |
|--------|-------------------|------|
| **Draft export (FastF1 in this repo)** | [`presentation/d/pull_race_laps.py`](pull_race_laps.py) → [`data/race_laps_japan_2026_pia_vs_ant.csv`](data/race_laps_japan_2026_pia_vs_ant.csv) | Reproducible prototype for Part D (`PIA`, `ANT` codes). Prefix slide assets `DRAFT_20260427_*.png` etc. |
| **Authoritative team extract (teammate B)** | **To be agreed** — e.g. `presentation/d/data/lap_times_team_B.csv` | **Presentation numbers** for slides + 1000-word group report once committed. |

## Scope definitions (Draft — align verbally with [`flow.md`](../../flow.md))

| Label | Lap window (`LapNumber` inclusive, both drivers unless noted) | Intent |
|-------|------------------------------------------------------------------|--------|
| **Scope A** | Full race (no min/max filter) | Baseline “naive” aggregate — includes SC, in/out laps, all stints. |
| **Scope B** | `19`–`53` | **Draft proxy** for *post–Oscar pit on lap 18* hard-stint discussion in `flow.md` — **not** a substitute for timing-app pit-out lap physics; team must confirm boundaries after B’s audit. |
| **Scope B alt** (optional) | `11`–`17` | Exploratory medium-tyre window before scripted pit window (Lando L16 / Oscar L18). |

**Slide rule:** Always caption figures with **`DRAFT_2026-04-27`**, scope letters, units (**s**, **s/lap**, **s cumulative**).

## Rules

1. **Freeze** the dataset used in the final deck at a named commit or dated filename once B publishes it.
2. **Do not overwrite** teammate B CSV with ad-hoc re-pulls silently — keep filename versions (`*_2026-05-07.csv`).
3. **`what_if_model.py`** numbers must trace to whichever dataset the team designates **final** — ingest via [`ingest_team_b.py`](ingest_team_b.py).
4. If FastF1 and B disagree on specific laps (interpolation, deleted laps), defer to **B’s audit** after manual spot-check against F1 Timing App screenshots / official PDF.

## Environment variable override

[`ingest_team_b.py`](ingest_team_b.py) reads **`F1_TEAM_B_LAPTIMES_CSV`**. Leave unset to fall back to the FastF1 export (`race_laps_japan_2026_pia_vs_ant.csv`).

Run [`parity_check.py`](parity_check.py) after swapping the env var:

```bash
cd /path/to/F1_final
uv run python presentation/d/parity_check.py
F1_TEAM_B_LAPTIMES_CSV=presentation/d/data/lap_times_team_B.csv uv run python presentation/d/parity_check.py
```
