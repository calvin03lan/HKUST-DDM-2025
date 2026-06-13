"""
依 `.cursor/skills/f1-structuring-skill/SKILL.md` 規則，對單圈遙測 CSV
追加 Corner_Type、Driving_State、Mini_Sector。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_CSV = "piastri_2026_japan_fastest_lap_telemetry.csv"
OUTPUT_CSV = "piastri_2026_japan_structured.csv"


def _to_bool_bike(s: pd.Series) -> pd.Series:
    if s.dtype == "bool" or s.dtype == np.bool_:
        return s
    if pd.api.types.is_bool_dtype(s):
        return s

    def one(x) -> object:
        if pd.isna(x):
            return x
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        st = str(x).strip()
        if st in ("True", "1", "1.0"):
            return True
        if st in ("False", "0", "0.0", ""):
            return False
        return x

    return s.map(one)


def structure_lap_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("Speed", "Throttle", "Brake"):
        if c not in out.columns:
            raise ValueError(f"必要欄位遺失: {c}")
    if "Distance" not in out.columns:
        raise ValueError("必要欄位遺失: Distance")

    out["Brake"] = _to_bool_bike(out["Brake"])
    for c in ("Speed", "Throttle", "Brake"):
        out[c] = out[c].replace("", np.nan)
        if c in ("Speed", "Throttle"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].ffill()

    s = out["Speed"].to_numpy(float)
    t = out["Throttle"].to_numpy(float)
    br = out["Brake"].to_numpy(bool)

    corner = np.select(
        [
            (s >= 250) & (t >= 95.0),
            ((s >= 200) & (s < 250)) | ((s >= 250) & (t < 95.0)),
            (s >= 120) & (s < 200),
        ],
        [
            "Straight",
            "High-Speed Corner",
            "Medium-Speed Corner",
        ],
        default="Low-Speed Corner",
    )
    out["Corner_Type"] = corner

    dstate = np.select(
        [
            (t >= 95.0) & (~br),
            (t < 5.0) & (br),
            (t > 5.0) & (br),
            (t < 5.0) & (~br),
        ],
        [
            "Full Acceleration",
            "Heavy Braking",
            "Trail Braking",
            "Coasting",
        ],
        default="Transition",
    )
    out["Driving_State"] = dstate

    d = out["Distance"].to_numpy(float)
    dmax = float(np.nanmax(d)) if d.size else 0.0
    if not np.isfinite(dmax) or dmax == 0.0:
        mini = np.array(["MS_1"] * len(out), dtype=object)
    else:
        idx = np.minimum(
            np.floor((d * 10.0) / (dmax + 1e-12)),
            9.0,
        ).astype(np.int32)
        mini = np.array([f"MS_{i + 1}" for i in idx], dtype=object)
    out["Mini_Sector"] = mini

    return out


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    res = structure_lap_telemetry(df)
    res.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} rows={len(res)} cols={list(res.columns)}")


if __name__ == "__main__":
    main()
