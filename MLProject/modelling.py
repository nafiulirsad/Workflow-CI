"""
Retraining entry point for the Bitcoin price prediction MLProject.

Designed to run head-less inside GitHub Actions:
  * tracking URI is taken from MLFLOW_TRACKING_URI, falling back to a local
    file store (./mlruns) so no MLflow server is required in CI;
  * the trained model is exported to MLProject/artifacts/model so the Docker
    build step has a deterministic path to copy from;
  * the resulting run id is written to run_id.txt for later workflow steps.
"""
import os
import json
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "namadataset_preprocessing")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "artifacts", "model")
EXPERIMENT_NAME = "Bitcoin_Price_Prediction_CI"
RUN_NAME = "RandomForest_CI"


def resolve_tracking_uri() -> str:
    """Use the configured tracking server when available, otherwise a local file store."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    local_store = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(local_store, exist_ok=True)
    return f"file://{local_store}"


def start_run():
    """Reuse the run created by `mlflow run`, or open a fresh one when called directly."""
    existing_run_id = os.getenv("MLFLOW_RUN_ID")
    if existing_run_id:
        print(f"Reusing MLflow run created by `mlflow run`: {existing_run_id}")
        run = mlflow.start_run(run_id=existing_run_id)
        mlflow.set_tag("mlflow.runName", RUN_NAME)
        return run
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=RUN_NAME)


def load_dataset():
    print(f"Loading preprocessed data from {DATA_DIR}...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    print(f"Train shape={X_train.shape}, Test shape={X_test.shape}")
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    tracking_uri = resolve_tracking_uri()
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.sklearn.autolog(log_models=True)

    X_train, X_test, y_train, y_test = load_dataset()

    with start_run() as run:
        print(
            f"Training RandomForestRegressor("
            f"n_estimators={args.n_estimators}, max_depth={args.max_depth})..."
        )
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2_score", r2)

        print(f"CI Retrain Results: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        # Export the model to a fixed path so the Docker build does not have to
        # guess where MLflow stored the run artifacts.
        if os.path.isdir(EXPORT_DIR):
            import shutil

            shutil.rmtree(EXPORT_DIR)
        os.makedirs(os.path.dirname(EXPORT_DIR), exist_ok=True)
        mlflow.sklearn.save_model(
            sk_model=model,
            path=EXPORT_DIR,
            input_example=X_train.head(2),
        )
        print(f"Model exported to {EXPORT_DIR}")

        run_id = run.info.run_id
        with open(os.path.join(SCRIPT_DIR, "run_id.txt"), "w") as fh:
            fh.write(run_id)
        with open(os.path.join(SCRIPT_DIR, "metrics.json"), "w") as fh:
            json.dump(
                {"run_id": run_id, "mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2},
                fh,
                indent=2,
            )
        print(f"MLflow run finished: {run_id}")


if __name__ == "__main__":
    main()
