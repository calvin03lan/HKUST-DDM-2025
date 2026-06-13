import os
import fastf1

# 1. 自动处理缓存目录
cache_dir = './f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

# 加载 2026 年上海大奖赛的排位赛 Session
# 注意：'Q' 代表 Qualifying
session = fastf1.get_session(2026, 'Shanghai', 'Q')
session.load()

if session.laps.empty:
    raise RuntimeError(
        "未获取到排位赛圈速数据。请检查网络后重试，"
        "或先确认该场次在 FastF1 数据源中已可用。"
    )

# 获取四位车手的最快圈并导出遥测数据
drivers = ["ANT", "HAM", "RUS", "LEC"]

for code in drivers:
    driver_laps = session.laps.pick_driver(code)
    if driver_laps.empty:
        print(f"{code} 未找到有效圈速，跳过导出。")
        continue

    fastest_lap = driver_laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry().add_distance()
    out_file = f"2026_Shanghai_Q_{code}_telemetry.csv"

    print(f"{code} 最快圈速: {fastest_lap['LapTime']}")
    telemetry.to_csv(out_file, index=False)
    print(f"{code} 数据已保存到: {out_file}")

print("四位车手排位赛遥测数据下载并保存完成。")