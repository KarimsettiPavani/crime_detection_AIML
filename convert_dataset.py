import pandas as pd

# Load datasets
stations_df = pd.read_csv("dataset1.csv")
crime_df = pd.read_csv("dataset2.csv")

# Get unique zones safely
zones = stations_df["zone"].dropna().astype(str).unique()

# Remove totalcrime if exists
if "totalcrime" in crime_df.columns:
    crime_df = crime_df.drop(columns=["totalcrime"])

# Clean year column
crime_df["year"] = pd.to_numeric(crime_df["year"], errors="coerce")

# Clean all crime columns properly
crime_columns = [col for col in crime_df.columns if col != "year"]

for col in crime_columns:
    crime_df[col] = (
        crime_df[col]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
    )
    crime_df[col] = pd.to_numeric(crime_df[col], errors="coerce")

# Drop invalid rows
crime_df = crime_df.dropna(subset=["year"])

# Convert wide to long
crime_long = crime_df.melt(
    id_vars=["year"],
    var_name="crime_type",
    value_name="cases"
)

# Drop invalid cases
crime_long = crime_long.dropna(subset=["cases"])

# Convert cases to numeric again (extra safety)
crime_long["cases"] = pd.to_numeric(crime_long["cases"], errors="coerce")

final_data = []

for _, row in crime_long.iterrows():
    for zone in zones:
        cases_value = float(row["cases"])
        distributed_value = cases_value / len(zones)

        final_data.append([
            int(row["year"]),
            zone,
            row["crime_type"],
            distributed_value
        ])

final_df = pd.DataFrame(
    final_data,
    columns=["year", "zone", "crime_type", "cases"]
)

final_df.to_csv("crime_data.csv", index=False)

print("✅ crime_data.csv created successfully!")