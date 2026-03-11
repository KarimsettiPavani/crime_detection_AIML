import pandas as pd

def get_high_risk_zone(model, df, le_zone, le_crime):

    future_predictions = []

    for zone in df["zone"].unique():
        for crime in df["crime_type"].unique():

            zone_val = le_zone.transform([zone])[0]
            crime_val = le_crime.transform([crime])[0]

            sample = pd.DataFrame({
                "year": [2026],
                "zone_encoded": [zone_val],
                "crime_encoded": [crime_val]
            })

            predicted = model.predict(sample)[0]
            future_predictions.append([zone, predicted])

    future_df = pd.DataFrame(
        future_predictions,
        columns=["Zone", "Predicted"]
    )

    zone_risk = future_df.groupby("Zone")["Predicted"].sum().reset_index()

    high_risk_zone = zone_risk.sort_values(
        by="Predicted",
        ascending=False
    ).iloc[0]["Zone"]

    return high_risk_zone, zone_risk