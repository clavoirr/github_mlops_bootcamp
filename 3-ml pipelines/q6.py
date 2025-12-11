from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import urllib.request
import os
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import joblib


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


# ---------------- Q5 + Q6 ---------------- #
@task
def train_and_log_model(df):

    # MLflow setup
    mlflow.set_experiment("NYC-Taxi-Homework-Experiment")

    with mlflow.start_run() as run:

        # ---------------- Train model ---------------- #
        features = ["PULocationID", "DOLocationID"]
        dicts = df[features].to_dict(orient="records")
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        y = df["duration"].values

        model = LinearRegression()
        model.fit(X, y)

        intercept = model.intercept_
        print(f"Q5 - Model Intercept: {intercept:.2f}")

        # ---------------- MLflow logging ---------------- #
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("vectorizer", "DictVectorizer")

        # Save DV + model in one file
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "model.pkl")
            joblib.dump((dv, model), filepath)
            mlflow.log_artifact(filepath, artifact_path="model")

        run_id = run.info.run_id
        print("MLflow run_id:", run_id)

    # ---------------- Retrieve MLmodel file size (Q6 answer) ---------------- #

    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, "model")

    model_size = 0
    for artifact in artifacts:
        local_path = client.download_artifacts(run_id, artifact.path)
        size = os.path.getsize(local_path)
        model_size += size

    print(f"Q6 - MLmodel file size (model_size_bytes): {model_size}")

    return model_size


@flow(name="NYC-Taxi-MLflow-Prefect-Pipeline")
def taxi_pipeline():
    df_raw = download_and_load()
    df_clean = prepare_data(df_raw)
    model_size = train_and_log_model(df_clean)
    return model_size


if __name__ == "__main__":
    taxi_pipeline()
