import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def train_and_predict():

    df = pd.read_csv("crime_data.csv")

    le_zone = LabelEncoder()
    le_crime = LabelEncoder()

    df["zone_encoded"] = le_zone.fit_transform(df["zone"])
    df["crime_encoded"] = le_crime.fit_transform(df["crime_type"])

    X = df[["year", "zone_encoded", "crime_encoded"]]
    y = df["cases"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model, df, le_zone, le_crime