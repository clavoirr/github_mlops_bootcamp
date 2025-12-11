import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    # Compute duration in minutes
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Filter trips between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to string
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    # Combine pickup and dropoff IDs
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    # Convert to list of dictionaries for DV
    dicts = df[categorical + numerical].to_dict(orient="records")

    # Fit or transform DictVectorizer
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@click.command()
@click.option("--raw_data_path", help="Folder with raw taxi parquet files")
@click.option("--dest_path", help="Folder where output files will be saved")
def run_data_prep(raw_data_path: str, dest_path: str, dataset="green"):

    # Load January, February, March 2023 data
    df_train = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet"))
    df_val   = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet"))
    df_test  = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet"))

    # Targets
    y_train = df_train["duration"].values
    y_val   = df_val["duration"].values
    y_test  = df_test["duration"].values

    # DictVectorizer
    dv = DictVectorizer()

    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _    = preprocess(df_val, dv, fit_dv=False)
    X_test, _   = preprocess(df_test, dv, fit_dv=False)

    # Create output folder
    os.makedirs(dest_path, exist_ok=True)

    # Save files
    dump_pickle(dv,                 os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val,   y_val),   os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test,  y_test),  os.path.join(dest_path, "test.pkl"))

    print("Saved 4 files to:", dest_path)


if __name__ == "__main__":
    run_data_prep()
