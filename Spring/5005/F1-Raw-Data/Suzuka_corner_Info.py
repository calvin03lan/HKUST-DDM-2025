import fastf1
import pandas as pd
import numpy as np

# 启用缓存以加速后续运行（推荐）
fastf1.Cache.enable_cache('fastf1_cache')

# 获取2026年日本大奖赛正赛的数据
session = fastf1.get_session(2026, 'Japan', 'R')
session.load(telemetry=True, weather=False)

# 获取最快圈，用于分析弯道速度和行驶轨迹
laps = session.laps
fast_lap = laps.pick_fastest()
telemetry = fast_lap.get_telemetry()

# 获取官方的赛道弯道信息（包含距离和编号）
circuit_info = session.get_circuit_info()

results = []
for index, corner in circuit_info.corners.iterrows():
    c_dist = corner['Distance']
    c_num = corner['Number']
    
    # 1. 速度分类 (Speed Class)
    # 取弯道前后50米范围内的最低速度作为该弯道的APEX速度
    mask = (telemetry['Distance'] >= c_dist - 50) & (telemetry['Distance'] <= c_dist + 50)
    t_corner = telemetry.loc[mask]
    
    if not t_corner.empty:
        min_speed = t_corner['Speed'].min()
        if min_speed < 130:
            speed_class = 'Low'
        elif min_speed < 200:
            speed_class = 'Medium'
        else:
            speed_class = 'High'
    else:
        min_speed = np.nan
        speed_class = 'Unknown'
        
    # 2. 弯道方向 (Direction)
    try:
        # 获取当前距离对应的数据索引
        idx_c = telemetry['Distance'].searchsorted(c_dist)
        # 选取弯道前后各10个点（避免波动带来误差）
        idx_b = max(0, idx_c - 10)
        idx_a = min(len(telemetry)-1, idx_c + 10)
        
        p_b = (telemetry.iloc[idx_b]['X'], telemetry.iloc[idx_b]['Y'])
        p_c = (telemetry.iloc[idx_c]['X'], telemetry.iloc[idx_c]['Y'])
        p_a = (telemetry.iloc[idx_a]['X'], telemetry.iloc[idx_a]['Y'])
        
        # 构造线段向量
        v1 = (p_c[0] - p_b[0], p_c[1] - p_b[1])
        v2 = (p_a[0] - p_c[0], p_a[1] - p_c[1])
        
        # 通过二维向量的叉乘判断转向：如果叉积 < 0 为右转，> 0 为左转
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        direction = 'Right' if cross < 0 else 'Left' 
    except Exception as e:
        direction = 'Unknown'

    results.append({
        'Corner ID': c_num, 
        'Distance (m)': round(c_dist, 2), 
        'Direction': direction, 
        'Min Speed (km/h)': round(min_speed, 2) if pd.notna(min_speed) else 'N/A', 
        'SpeedClass': speed_class
    })

# 转换为DataFrame进行展示
df = pd.DataFrame(results)

# 构建 Markdown 表格并输出到文件
md_content = "# Suzuka Circuit Corners\n\n"
md_content += "| ID | Distance (m) | Direction | Min Speed (km/h) | Speed Class |\n"
md_content += "| --- | --- | --- | --- | --- |\n"
for _, row in df.iterrows():
    min_speed_str = f"{row['Min Speed (km/h)']:.2f}" if isinstance(row['Min Speed (km/h)'], (int, float)) else str(row['Min Speed (km/h)'])
    md_content += f"| {row['Corner ID']} | {row['Distance (m)']:.2f} | {row['Direction']} | {min_speed_str} | {row['SpeedClass']} |\n"

with open('Suzuka_corner_Info.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print(md_content)
print("\n结果已成功输出到 Suzuka_corner_Info.md 文件中！")