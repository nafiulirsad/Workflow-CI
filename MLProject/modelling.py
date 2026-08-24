import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Set tracking URI to local
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bitcoin_Price_Prediction_CI")

    # Enable autolog
    mlflow.sklearn.autolog()

    # Load preprocessed dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "namadataset_preprocessing")

    print(f"Loading preprocessed data from {data_dir}...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Train model in MLflow run
    with mlflow.start_run(run_name="RandomForest_CI"):
        print(f"Training RandomForestRegressor(n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"CI Retrain Results: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()
