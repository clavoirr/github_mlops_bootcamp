# train.py
import os
import tempfile
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient

DATA_PATH = "yellow_tripdata_2023-03.parquet"
MLFLOW_EXPERIMENT_NAME = "yellow_taxi_mar2023_prefect_homework"

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    # Q3: print number of loaded records
    print("Q3 - raw rows loaded:", len(df))

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def train_and_log(df):
    # Q4: print number after filtering/prep
    print("Q4 - rows after preparation:", len(df))

    # features
    categorical = ['PULocationID', 'DOLocationID']
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    def prepare(df_):
        df = df_.copy()
        dicts = df[categorical].to_dict(orient='records')
        return dicts, df.duration.values

    X_train_dicts, y_train = prepare(df_train)
    X_val_dicts, y_val = prepare(df_val)

    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train_dicts)
    X_val = dv.transform(X_val_dicts)

    model = LinearRegression()
    model.fit(X_train, y_train)

    intercept = float(model.intercept_)
    print("Q5 - model intercept:", round(intercept, 2))

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer", "DictVectorizer")
        mlflow.log_param("n_train", len(X_train_dicts))
        mlflow.log_param("n_val", len(X_val_dicts))

        # Save model + dv to a temp folder and log as an artifact
        tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(tmpdir, "model.npz")
        # Save with numpy: store intercept, coef, feature names and dv
        np.savez(model_path, intercept=model.intercept_, coef=model.coef_, feature_names=dv.feature_names_)
        mlflow.log_artifact(model_path, artifact_path="model")  # will create artifacts/model/model.npz

        # Also use mlflow.sklearn.log_model for standard MLflow model directory (optional)
        import joblib
        sklearn_model_path = os.path.join(tmpdir, "sklearn_model.pkl")
        joblib.dump({"model": model, "dv": dv}, sklearn_model_path)
        mlflow.log_artifact(sklearn_model_path, artifact_path="model")

        run_id = run.info.run_id
        print("MLflow run_id:", run_id)

    # Retrieve artifact size for the MLModel file (or the artifact file) using MlflowClient
    client = MlflowClient()
    # List artifacts under 'model'
    artifacts = client.list_artifacts(run_id, path="model")
    # We'll find the first artifact and download it to measure file size
    total_size = 0
    for artifact in artifacts:
        # download_artifacts returns a local path
        local_path = client.download_artifacts(run_id, artifact.path, dst_path=tempfile.mkdtemp())
        if os.path.isfile(local_path):
            size = os.path.getsize(local_path)
            # print each artifact and its size
            print(f"artifact {artifact.path} size_bytes = {size}")
            total_size += size
        else:
            # if local_path is a dir, walk it
            dir_size = 0
            for root, _, files in os.walk(local_path):
                for f in files:
                    fp = os.path.join(root, f)
                    dir_size += os.path.getsize(fp)
            print(f"artifact {artifact.path} total_dir_size = {dir_size}")
            total_size += dir_size

    # Print the model_size_bytes (sum of artifact sizes under 'model')
    print("Q6 - model_size_bytes (sum of artifacts under model/):", total_size)
    return intercept, total_size

def main():
    df = read_dataframe(DATA_PATH)
    train_and_log(df)

if __name__ == "__main__":
    main()
