import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle
from lightgbm import LGBMRegressor
import os


# Load data:
def load_data(train_url: str, test_url: str) -> tuple:
    try:
        train_df = pd.read_csv(train_url)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training file not found: {train_url}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse training CSV: {e}")

    try:
        test_df = pd.read_csv(test_url)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test file not found: {test_url}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse test CSV: {e}")

    if "time" not in train_df.columns or "time" not in test_df.columns:
        raise KeyError("Target column 'time' missing from input data")

    X_train = train_df.drop(columns=["time"])
    y_train = train_df["time"]

    X_test = test_df.drop(columns=["time"])
    y_test = test_df["time"]

    if X_train.empty or X_test.empty:
        raise ValueError("Feature matrix is empty")

    return X_train, X_test, y_train, y_test

# load model:
def load_model(model_url: str) -> LGBMRegressor:
    try:
        with open(model_url, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_url}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    if not hasattr(model, "predict"):
        raise TypeError("Loaded object is not a valid trained model")

    return model

# Evaluate model performance
def evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   model: LGBMRegressor) -> dict:

    try:
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
    except Exception as e:
        raise RuntimeError(f"Prediction or metric computation failed: {e}")

    try:
        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="neg_mean_absolute_error"
        )
        neg_mean_absolute_error = scores.mean()
    except ValueError as e:
        raise ValueError(f"Cross-validation failed: {e}")

    metrics = {
        "train_mean_absolute_error": train_mae,
        "train_r2_score": train_r2,
        "test_mean_absolute_error": test_mae,
        "test_r2_score": test_r2,
        "neg_mean_absolute_error": neg_mean_absolute_error
    }

    return metrics

# save metrics in reports:
def save_metrics(metrics: dict, url: str) -> None:
    try:
        os.makedirs(os.path.dirname(url), exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create reports directory: {e}")

    try:
        with open(url, "w") as file:
            json.dump(metrics, file, indent=4)
    except IOError as e:
        raise IOError(f"Failed to save metrics to {url}: {e}")

def main() -> None:
    try:
        train_url = "data/processed/train_processed.csv"
        test_url = "data/processed/test_processed.csv"

        X_train, X_test, y_train, y_test = load_data(train_url, test_url)

        model_url = "models/lgbm_regressor.pkl"
        model = load_model(model_url)

        metrics = evaluate_model(X_train, X_test, y_train, y_test, model)

        metrics_url = "reports/metrics.json"
        save_metrics(metrics, metrics_url)

    except FileNotFoundError as e:
        raise RuntimeError(f"[FILE ERROR] {e}")

    except KeyError as e:
        raise RuntimeError(f"[SCHEMA ERROR] {e}")

    except ValueError as e:
        raise RuntimeError(f"[DATA / CONFIG ERROR] {e}")

    except RuntimeError:
        raise  # preserve original traceback

    except Exception as e:
        raise RuntimeError(f"[UNEXPECTED ERROR] {e}")


if __name__ == "__main__":
    main()