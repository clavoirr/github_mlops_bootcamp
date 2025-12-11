import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # -------------------------
    # MLflow Setup
    # -------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("random-forest-experiment")

    # Enable autologging
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        # NOTE: Your code does NOT specify `min_samples_split`,
        # so RandomForestRegressor uses the DEFAULT = **2**
        rf = RandomForestRegressor(
            max_depth=10,
            random_state=0
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)

        # manual logging (optional because autolog logs metrics too)
        mlflow.log_metric("rmse_manual", rmse)

        print("RMSE:", rmse)


if __name__ == '__main__':
    run_train()
