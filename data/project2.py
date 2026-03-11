#crime master dataset creation
import pandas as pd
import numpy as np

# Load datasets
crime_df = pd.read_csv(r"C:\Users\user\Downloads\vizag_district_crime.csv")
stations_df = pd.read_csv(r"C:\Users\user\Downloads\vizag_police_stations.csv")

crime_types = ["murder", "kidnapping", "theft", "robbery", "human_trafficking"]

master_data = []

for _, row in crime_df.iterrows():
    year = row["year"]
    
    for month in range(1, 13):
        for _, station in stations_df.iterrows():
            for crime in crime_types:
                
                # Distribute yearly crime randomly across months & stations
                yearly_cases = row[crime]
                monthly_station_cases = np.random.poisson(
                    yearly_cases / (12 * len(stations_df))
                )
                
                master_data.append([
                    year,
                    month,
                    station["zone"],
                    station["station_name"],
                    crime,
                    monthly_station_cases,
                    station["latitude"],
                    station["longitude"]
                ])

master_df = pd.DataFrame(master_data, columns=[
    "year",
    "month",
    "zone",
    "police_station",
    "crime_type",
    "cases",
    "latitude",
    "longitude"
])

master_df.to_csv("vizag_crime_master.csv", index=False)

print("Master dataset created successfully!")
print(master_df.head())