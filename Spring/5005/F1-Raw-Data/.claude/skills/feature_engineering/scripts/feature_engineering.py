import fastf1
import pandas as pd
import numpy as np
import os

# 参数设置
YEAR = 2026
RACE = 'Japan'
SESSION = 'R'
DATA_DIR = 'structured_data/'
CSV_IN = os.path.join(DATA_DIR, 'structured_data.csv')
CSV_OUT = os.path.join(DATA_DIR, 'structured_data_with_corners.csv')
CORNER_INFO_OUT = os.path.join(DATA_DIR, 'corner_info.csv')

# 1. 下载和获取基础赛道遥测数据
fastf1.Cache.enable_cache('fastf1_cache')
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load(telemetry=True, weather=False)

laps = session.laps
fast_lap = laps.pick_fastest()
telemetry = fast_lap.get_telemetry()
circuit_info = session.get_circuit_info()

results = []
for index, corner in circuit_info.corners.iterrows():
    c_dist = corner['Distance']
    c_num = corner['Number']
    
    # 获取弯心前后的遥测数据
    t_corner = telemetry[(telemetry['Distance'] >= c_dist - 150) & (telemetry['Distance'] <= c_dist + 150)]
    
    if not t_corner.empty:
        # Min speed near corner (within 50 meters of apex)
        apex_data = t_corner[(t_corner['Distance'] >= c_dist - 50) & (t_corner['Distance'] <= c_dist + 50)]
        min_speed = apex_data['Speed'].min() if not apex_data.empty else np.nan
        
        # 速度阈值分类规则
        if pd.isna(min_speed):
            speed_class = 'Unknown'
        elif min_speed <= 120:
            speed_class = 'Low'
        elif min_speed < 200:
            speed_class = 'Medium'
        else:
            speed_class = 'High'
            
        # 入弯和出弯点简单估算：
        # 对于低速/中速弯，入弯点定义为弯前刹车开始点或速度下降点（简单化：Apex - 100）
        # 出弯点定义为回到全油门点或直道点（简单化：Apex + 50）
        # 这里用一种启发式固定距离作为示例，可以进一步用导数/加速度找极值。
        entry_dist = c_dist - 100 if speed_class != 'High' else c_dist - 50
        exit_dist = c_dist + 50 
    else:
        min_speed = np.nan
        speed_class = 'Unknown'
        entry_dist = c_dist - 50
        exit_dist = c_dist + 50
        
    # 方向判断
    try:
        idx_c = telemetry['Distance'].searchsorted(c_dist)
        idx_b = max(0, idx_c - 10)
        idx_a = min(len(telemetry)-1, idx_c + 10)
        p_b = (telemetry.iloc[idx_b]['X'], telemetry.iloc[idx_b]['Y'])
        p_c = (telemetry.iloc[idx_c]['X'], telemetry.iloc[idx_c]['Y'])
        p_a = (telemetry.iloc[idx_a]['X'], telemetry.iloc[idx_a]['Y'])
        v1 = (p_c[0] - p_b[0], p_c[1] - p_b[1])
        v2 = (p_a[0] - p_c[0], p_a[1] - p_c[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        direction = 'Right' if cross < 0 else 'Left'
    except:
        direction = 'Unknown'

    results.append({
        'Corner_ID': c_num, 
        'Apex_Distance': c_dist, 
        'Direction': direction, 
        'Min_Speed': min_speed, 
        'Speed_Class': speed_class,
        'Entry_Distance': entry_dist,
        'Exit_Distance': exit_dist
    })

corner_df = pd.DataFrame(results)
corner_df.to_csv(CORNER_INFO_OUT, index=False)
print(f"Corner mapping saved to {CORNER_INFO_OUT}")

# 2. 对 structured_data.csv 进行映射
if os.path.exists(CSV_IN):
    data = pd.read_csv(CSV_IN)
    data['Corner_ID'] = 0  # 0表示不在弯道（直道）
    data['Corner_Phase'] = 'Straight'
    data['Corner_Direction'] = 'Straight'
    data['Corner_SpeedClass'] = 'Straight'
    
    # 根据之前统计的弯道距离范围赋值
    for _, row in corner_df.iterrows():
        cid = row['Corner_ID']
        entry_d = row['Entry_Distance']
        apex_d = row['Apex_Distance']
        exit_d = row['Exit_Distance']
        direction = row['Direction']
        speed_class = row['Speed_Class']
        
        # 判断处在弯道区间
        mask = (data['Distance'] >= entry_d) & (data['Distance'] <= exit_d)
        data.loc[mask, 'Corner_ID'] = cid
        data.loc[mask, 'Corner_Direction'] = direction
        data.loc[mask, 'Corner_SpeedClass'] = speed_class
        
        # 细分阶段: Entry (Entry_Distance ~ Apex_Distance) / Exit (Apex_Distance ~ Exit_Distance)
        mask_entry = mask & (data['Distance'] <= apex_d)
        mask_exit = mask & (data['Distance'] > apex_d)
        
        data.loc[mask_entry, 'Corner_Phase'] = 'Entry'
        data.loc[mask_exit, 'Corner_Phase'] = 'Exit'

    data.to_csv(CSV_OUT, index=False)
    print(f"Structured data with corners saved to {CSV_OUT}")
else:
    print(f"File not found: {CSV_IN}")
