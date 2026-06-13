---
name: data-structuring
description: Processes CSV and JSON data files in the current project. Use when a user initiates a "Structure the Raw Data" command or use natural language to request data structuring.
---

In the current project, telemetry data is stored in CSV files containing 'telemetry' and JSON files containing 'lap_summary' in their names, which are both in the "raw_data" folder. To facilitate analysis and modeling, need to structure this raw data. The specific steps are as follows:

# Telemetry Data Cleaning
1. Time Format Unification: Convert the time formats of the `SessionTime` and `Time` columns into "hh:mm:ss.xxx", retaining three decimal places to ensure consistency and readability of the time data.
2. Column Sorting and Processing: Delete the `Date` column; perform only format conversion on the `SessionTime` column; keep the columns `Time`, `RPM`, `Speed`, `nGear`, `Throttle`, `Brake`, `DRS`, `Source`, `Distance`, `RelativeDistance`, `Status`, `X`, `Y`, and `Z` unchanged.
3. Missing Value Check: Perform statistical analysis on missing values for all columns and generate a missing value table, including column names, whether data is missing, the count, and the proportion of missing values. Provide processing recommendations based on the significance of each column.
4. Outlier Check: Perform outlier detection on numerical and boolean columns (such as `RPM`, `Speed`, `nGear`, `Throttle`, `Brake`, etc.) to identify and record values that fall outside reasonable ranges or do not conform to column definitions. Generate an outlier table and provide processing recommendations based on the significance of each column.
5. Save the missing value and outlier tables as `missing_and_outliers_report.csv`, then process the missing and outlying data according to the recommendations to generate `cleaned_telemetry.csv`.

# Data Unification
1. Data Broadcasting: Broadcast low-frequency data from `lap_summary.json` (including `Compound`, `TyreLife`, `TrackStatus`, and all secondary entries under `Weather`) to every row of `cleaned_telemetry.csv` using a "full fill" method, so that every millisecond-level record carries environmental context.
2. Sector Matching: Use `SessionTime` for interval determination. By calculating the cumulative values of `Sector1Time`, `Sector2Time`, and `Sector3Time`, precisely label the telemetry data as Sector 1, 2, or 3, ensuring every record corresponds to the correct track segment information.
3. New File Naming: Save the processed data as `structured_data.csv` to ensure the filename clearly reflects its content and purpose.
4. Save the generated scripts and CSV，JSON files to the "structured_data" folder in the project directory.