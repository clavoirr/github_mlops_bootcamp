from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import urllib.request
import os


URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
FILENAME = "yellow_tripdata_2023-03.parquet"


# ---------------- Q3 ---------------- #
@task
def download_and_load():
    if not os.path.exists(FILENAME):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, FILENAME)
    else:
        print("Dataset already exists.")

    df = pd.read_parquet(FILENAME)
    print(f"Q3 - Number of records loaded: {len(df)}")
    return df


# ---------------- Q4 ---------------- #
@task
def prepare_data(df):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)

    print(f"Q4 - Rows after preparation: {len(df)}")
    return df


# ---------------- Q5 ---------------- #
@task
def train_model(df):
    features = ["PULocationID", "DOLocationID"]

    dicts = df[features].to_dict(orient="records")
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    y = df["duration"].values

    model = LinearRegression()
    model.fit(X, y)

    intercept = model.intercept_

    print(f"Q5 - Model Intercept: {intercept:.2f}")

    return dv, model, intercept


# ---------------- FLOW ---------------- #
@flow(name="NYC-Taxi-Training-Pipeline")
def taxi_training_pipeline():
    df_raw = download_and_load()
    df_clean = prepare_data(df_raw)
    dv, model, intercept = train_model(df_clean)

    return intercept


if __name__ == "__main__":
    taxi_training_pipeline()
