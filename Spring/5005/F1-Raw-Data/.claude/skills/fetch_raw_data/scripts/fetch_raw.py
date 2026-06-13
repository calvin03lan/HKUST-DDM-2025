import fastf1
import os
import json

# Create a cache directory to speed up future requests
cache_dir = './fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable the cache
fastf1.Cache.enable_cache(cache_dir)

def main():
    print("Loading 2026 Japan GP Race session...")
    # Load the session data
    session = fastf1.get_session(2026, 'Japan', 'Race')
    session.load()

    print("Extracting Lando Norris's fastest lap...")
    # Get Norris's fastest lap
    nor_laps = session.laps.pick_driver('NOR')
    fastest_lap = nor_laps.pick_fastest()

    print("Retrieving telemetry data...")
    # Get the telemetry data for that specific lap
    telemetry = fastest_lap.get_telemetry()

    # Save the data to a CSV file
    output_filename = 'nor_jpn_2026_telemetry.csv'
    telemetry.to_csv(output_filename, index=False)
    print(f"Success! Fast lap telemetry data saved to {output_filename}")

    print("Extracting lap summary (Weather, Tires, Track Status, Sectors)...")
    # Get native weather data
    weather_data = fastest_lap.get_weather_data()
    
    # Collect lap-level metrics natively without modifying the original telemetry dataset
    lap_summary = {
        "Compound": str(fastest_lap.get('Compound', '')),
        "TyreLife": float(fastest_lap.get('TyreLife', 0.0)),
        "Sector1Time": str(fastest_lap.get('Sector1Time', '')),
        "Sector2Time": str(fastest_lap.get('Sector2Time', '')),
        "Sector3Time": str(fastest_lap.get('Sector3Time', '')),
        "TrackStatus": str(fastest_lap.get('TrackStatus', '')),
        "Weather": {
            "AirTemp": float(weather_data.get('AirTemp', 0.0)),
            "TrackTemp": float(weather_data.get('TrackTemp', 0.0)),
            "Humidity": float(weather_data.get('Humidity', 0.0)),
            "Pressure": float(weather_data.get('Pressure', 0.0)),
            "WindSpeed": float(weather_data.get('WindSpeed', 0.0)),
            "WindDirection": float(weather_data.get('WindDirection', 0.0)),
            "Rainfall": bool(weather_data.get('Rainfall', False))
        }
    }

    # Save summary as an independent JSON file
    summary_filename = 'nor_jpn_2026_lap_summary.json'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(lap_summary, f, indent=4, ensure_ascii=False)
    print(f"Success! Lap summary data saved to {summary_filename}")

if __name__ == '__main__':
    main()
