"""
使用 FastF1 下載指定期別、站別、車手「正賽中單圈最快圈」的合併遙測，並匯出 CSV。
（若要改為排位賽的單圈最快，請將 SESSION 改為 "Q"）
"""

from __future__ import annotations

from pathlib import Path

import fastf1

# --- 依需求修改 ---
YEAR = 2026
EVENT = "Japan"  # 日本大獎賽
SESSION = "R"  # 正賽；此處取正賽中該車手「單圈時間最短」的那一圈之遙測
DRIVER = "PIA"  # Oscar Piastri
OUT_CSV = Path(__file__).resolve().parent / "piastri_2026_japan_fastest_lap_telemetry.csv"


def main() -> None:
    cache_dir = Path(__file__).resolve().parent / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(YEAR, EVENT, SESSION)
    session.load(telemetry=True, laps=True)

    laps = session.laps.pick_drivers(DRIVER)
    if laps is None or laps.empty:
        raise SystemExit(
            f"此場次找不到車手 {DRIVER} 的圈次資料，請確認賽曆/車手代碼/該站是否已比賽。"
        )

    fastest = laps.pick_fastest()
    if fastest is None or fastest.empty:
        raise SystemExit(f"找不到 {DRIVER} 在本場的「單圈最快圈」可解析資料。")

    telem = fastest.get_telemetry()
    if hasattr(telem, "add_distance"):
        telem = telem.add_distance()

    telem.to_csv(OUT_CSV, index=False)
    print(f"已寫入：{OUT_CSV}（列數 {len(telem)}）")


if __name__ == "__main__":
    main()
