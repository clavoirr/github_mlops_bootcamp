import pandas as pd
import urllib.request
import os
from prefect import flow, task

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
FILENAME = "yellow_tripdata_2023-03.parquet"


@task
def download_data():
    if not os.path.exists(FILENAME):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, FILENAME)
    else:
        print("Dataset already exists.")
    return FILENAME


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task
def prepare_data(filename):
    df = read_dataframe(filename)
    print("Q4 - Size after data preparation:", len(df))
    return len(df)


@flow(name="Q4-Prefect-Data-Prep-Flow")
def prefect_data_flow():
    filename = download_data()
    final_size = prepare_data(filename)
    return final_size


if __name__ == "__main__":
    prefect_data_flow()
