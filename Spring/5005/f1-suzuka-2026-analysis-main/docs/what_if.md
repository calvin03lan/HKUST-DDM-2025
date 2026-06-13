# What-If: Safety Car & gap projection (Part D)

Pedagogical scaffold for the **McLaren 2026 Japanese GP** narrative. Treat all numbers as **hypothesis inputs** until teammate **B** publishes authoritative lap-time extracts.

## 1. Symbols

| Symbol | Meaning | Unit |
|--------|---------|------|
| $G_0$ | Gap (Piastri ahead **positive**) at reference event (e.g. SC deployment or hypothetical freeze lap) | s |
| $N$ | Remaining racing laps **after** that reference until chequered flag | laps |
| $T_{\mathrm{PIA}}$ | Mean stint lap time for Piastri in clean air after last relevant stop | s |
| $T_{\mathrm{OPP}}$ | Mean stint lap time for labelled opponent stint (e.g. Antonelli on older rubber) | s |
| $\Delta P = T_{\mathrm{OPP}} - T_{\mathrm{PIA}}$ | **Pace delta** (`>0` ⇒ Piastri faster per lap) | s/lap |

Optional tire-degradation slopes (advanced):

| Symbol | Meaning |
|--------|---------|
| $d_{\mathrm{PIA}},\, d_{\mathrm{OPP}}$ | Extra seconds per lap vs lap index inside stint (linear approx.) |

## 2. Tier 0 — linear projection

**Assumption set A (explicit on slides)**:

- Ignore traffic / blue flags; stationary **random undercut/overcut reshuffles**.
- Ignore fuel-mass curvature; approximate **constant** differential across $N$.
- Single opponent focus; multisector SC bunching boiled down to **one scalar** $G_0$.

$$
G_{\mathrm{final,\,simple}} = G_0 + \Delta P \cdot N .
$$

Interpretation:

- $\Delta P > 0$ and $G_{\mathrm{final}} > 0$ ⇒ projected **Piastri lead** at finish under these assumptions.

## 3. Optional — cumulative degradation-aware sum

Discrete loop (more defensible than a closed form if slopes differ):

For lap index $k = 1\dots N$,

$$
\Delta P_k = \big(T_{\mathrm{OPP}} + d_{\mathrm{OPP}}\cdot k\big) - \big(T_{\mathrm{PIA}} + d_{\mathrm{PIA}}\cdot k\big),
\qquad G_{\mathrm{final}} = G_0 + \sum_{k=1}^{N} \Delta P_k .
$$

Use only when B supplies believable stint windows and degradation estimates **or** Monte-Carlo sensitivities shown.

## 4. Data layers (cannot mix)

| Layer | Typical source | Suitable for What-If? |
|-------|----------------|------------------------|
| Single-lap merged telemetry CSV (assignment) | `piastri_2026_*_telemetry.csv` | **No** for $G_{\mathrm{final}}$ — micro signals only |
| Lap table | `pull_race_laps.py`, teammate B CSV | **Yes** once aligned (`sync_with_team_B.md`) |
| Narrative quotes | Primary press / team media | Supporting text only |

## 5. Scope A vs Scope B (draft operationalisation)

**Version tag:** `DRAFT_2026-04-27` (see [`sync_with_team_B.md`](sync_with_team_B.md)).

| Scope | Lap filter on `LapNumber` | Use case |
|-------|---------------------------|----------|
| **A** | Entire race $(1\ldots L_{\max})$ | Shows how “blind average” pace can misalign with `flow.md` stint story. |
| **B** | $19\ldots 53$ (exploratory) | Coarse **post–pit (lap 18 script)** window for **PIA**/**ANT** — team + B must validate; not proof of clean air. |

Recomputing means when switching A→B is **data-centric** (same model tier, different row scope).

## 6. Sensitivity checklist

Report at least:

- $(G_0,\, \Delta P,\, N)$ perturbations $\pm10\%$ on $\Delta P$ once B locks means.
- SC timing sensitivity: classify whether $G_0$ flips sign under conservative vs optimistic reconstructions.

Automated draft report: [`run_draft_insight.py`](run_draft_insight.py).
