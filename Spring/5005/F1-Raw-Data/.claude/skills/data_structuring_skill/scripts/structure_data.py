import pandas as pd
import json
import numpy as np

def format_timedelta(td):
    if pd.isnull(td):
        return np.nan
    try:
        # Convert to pd.Timedelta if not already
        td = pd.to_timedelta(td)
        total_seconds = td.total_seconds()
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    except:
        return td

def main():
    print("Loading data...")
    df = pd.read_csv('nor_jpn_2026_telemetry.csv')
    with open('nor_jpn_2026_lap_summary.json', 'r') as f:
        summary = json.load(f)

    # 1. Telemetry Data Cleaning
    # 1.2 Column Sorting and Processing: Delete 'Date'.
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    
    # Keep specific columns
    cols_to_keep = ['SessionTime', 'Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source', 'Distance', 'RelativeDistance', 'Status', 'X', 'Y', 'Z']
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # 1.3 Missing Value Check
    missing_stats = df.isnull().sum().reset_index()
    missing_stats.columns = ['Column', 'Missing_Count']
    missing_stats['Missing_Proportion'] = missing_stats['Missing_Count'] / len(df)
    missing_stats['Type'] = 'Missing'
    missing_stats['Recommendation'] = np.where(missing_stats['Missing_Count'] > 0, 'Forward fill or interpolate based on time', 'No action needed')

    # 1.4 Outlier Check
    numerical_cols = ['RPM', 'Speed', 'nGear', 'Throttle']
    outlier_records = []
    
    for col in numerical_cols:
        if col in df.columns:
            # Simple bounds for F1 telemetry
            if col == 'RPM':
                outliers = df[(df[col] < 0) | (df[col] > 15000)]
            elif col == 'Speed':
                outliers = df[(df[col] < 0) | (df[col] > 400)]
            elif col == 'nGear':
                outliers = df[(df[col] < 0) | (df[col] > 8)]
            elif col == 'Throttle':
                outliers = df[(df[col] < 0) | (df[col] > 100)]
            
            count = len(outliers)
            if count > 0:
                outlier_records.append({
                    'Column': col,
                    'Outlier_Count': count,
                    'Type': 'Outlier',
                    'Recommendation': 'Cap to logical min/max or interpolate'
                })

    outlier_df = pd.DataFrame(outlier_records)
    if not outlier_df.empty:
        report_df = pd.concat([missing_stats, outlier_df], ignore_index=True)
    else:
        report_df = missing_stats

    # 1.5 Save the report
    report_df.to_csv('missing_and_outliers_report.csv', index=False)
    print("Saved missing_and_outliers_report.csv")

    # Process missing and outliers (Execute recommendations)
    # Fill missing values
    df = df.ffill().bfill() # Forward fill then backward fill
    
    # Cap outliers
    if 'RPM' in df.columns: df['RPM'] = df['RPM'].clip(0, 15000)
    if 'Speed' in df.columns: df['Speed'] = df['Speed'].clip(0, 400)
    if 'nGear' in df.columns: df['nGear'] = df['nGear'].clip(0, 8)
    if 'Throttle' in df.columns: df['Throttle'] = df['Throttle'].clip(0, 100)

    # 1.1 Time Format Unification (Do this after filling missing to avoid NaNs)
    df['SessionTime'] = pd.to_timedelta(df['SessionTime']).apply(format_timedelta)
    df['Time'] = pd.to_timedelta(df['Time']).apply(format_timedelta)

    df.to_csv('cleaned_telemetry.csv', index=False)
    print("Saved cleaned_telemetry.csv")

    # 2. Data Unification
    # Create SessionTime objects for sector calculation
    df_parse = pd.read_csv('nor_jpn_2026_telemetry.csv')
    session_times = pd.to_timedelta(df_parse['SessionTime'])
    
    # 2.1 Data Broadcasting
    df['Compound'] = summary.get('Compound', '')
    df['TyreLife'] = summary.get('TyreLife', 0)
    df['TrackStatus'] = summary.get('TrackStatus', '')
    
    weather = summary.get('Weather', {})
    for k, v in weather.items():
        df[f'Weather_{k}'] = v
        
    # 2.2 Sector Matching
    s1_time = pd.to_timedelta(summary.get('Sector1Time', '0 days 00:00:00'))
    s2_time = pd.to_timedelta(summary.get('Sector2Time', '0 days 00:00:00'))
    s3_time = pd.to_timedelta(summary.get('Sector3Time', '0 days 00:00:00'))
    
    # We need the lap start time. The first row's SessionTime is roughly the lap start.
    lap_start = session_times.iloc[0]
    s1_end = lap_start + s1_time
    s2_end = s1_end + s2_time
    
    sectors = []
    for st in session_times:
        if st <= s1_end:
            sectors.append("Sector 1")
        elif st <= s2_end:
            sectors.append("Sector 2")
        else:
            sectors.append("Sector 3")
            
    df['Sector'] = sectors

    # 2.3 New File Naming
    df.to_csv('structured_data.csv', index=False)
    print("Saved structured_data.csv")

if __name__ == '__main__':
    main()