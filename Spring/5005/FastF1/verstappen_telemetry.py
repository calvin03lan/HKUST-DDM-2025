import fastf1 as ff1
import matplotlib.pyplot as plt
import pandas as pd

# Enable the cache
ff1.Cache.enable_cache('cache')

# Load the session data
session = ff1.get_session(2026, 'Australia', 'R')
session.load()

# Get the laps for Max Verstappen
ver_laps = session.laps.pick_driver('VER')

# Get the fastest lap
fastest_lap = ver_laps.pick_fastest()

# Get the telemetry for the fastest lap
telemetry = fastest_lap.get_car_data().add_distance()

# Save the lap data to a CSV file
ver_laps.to_csv('verstappen_lap_data.csv')
print("Lap data saved to verstappen_lap_data.csv")


# Create the plot
fig, ax = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle(f'{session.event["EventName"]} {session.event.year} - Max Verstappen - Fastest Lap Telemetry')

ax[0].plot(telemetry['Distance'], telemetry['Speed'], label='Speed')
ax[0].set_xlabel('Distance in m')
ax[0].set_ylabel('Speed in km/h')
ax[0].legend()

ax[1].plot(telemetry['Distance'], telemetry['Throttle'], label='Throttle')
ax[1].set_xlabel('Distance in m')
ax[1].set_ylabel('Throttle (%)')
ax[1].legend()

ax[2].plot(telemetry['Distance'], telemetry['Brake'], label='Brake')
ax[2].set_xlabel('Distance in m')
ax[2].set_ylabel('Brake')
ax[2].legend()

# Save the plot
plt.savefig('verstappen_fastest_lap_telemetry.png')

print("Telemetry graph saved as verstappen_fastest_lap_telemetry.png")
