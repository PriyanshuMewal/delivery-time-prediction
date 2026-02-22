import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle
from lightgbm import LGBMRegressor
import os
import logging

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


# Load data:
def load_data(train_url: str, test_url: str) -> tuple:
    logger.info(f"Loading training data from {train_url} and test data from {test_url}")

    try:
        train_df = pd.read_csv(train_url)
    except FileNotFoundError:
        logger.error(f"Training file not found: {train_url}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse training CSV: {e}")
        raise

    try:
        test_df = pd.read_csv(test_url)
    except FileNotFoundError:
        logger.error(f"Test file not found: {test_url}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse test CSV: {e}")
        raise

    if "time" not in train_df.columns or "time" not in test_df.columns:
        logger.error("Either training or test data doesn't contain target column.")
        raise KeyError("Target column 'time' missing from input data")

    X_train = train_df.drop(columns=["time"])
    y_train = train_df["time"]

    X_test = test_df.drop(columns=["time"])
    y_test = test_df["time"]

    if X_train.empty or X_test.empty:
        logger.error("Either training or test data empty, check you preprocessing pipelines.")
        raise ValueError("Feature matrix is empty")

    logger.info("Dataset loaded successfully.")
    return X_train, X_test, y_train, y_test

# load model:
def load_model(model_url: str) -> LGBMRegressor:
    logger.info(f"Loading lightgbm model from {model_url}.")

    try:
        with open(model_url, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_url}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    if not hasattr(model, "predict"):
        logger.error("model doesn't contain predict method check the model building stage.")
        raise TypeError("Loaded object is not a valid trained model")

    logger.info("Model loaded successfully.")
    return model

# Evaluate model performance
def evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   model: LGBMRegressor) -> dict:
    logger.info("Evaluating model performance ...")

    try:
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"Prediction or metric computation failed: {e}")
        raise

    try:
        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="neg_mean_absolute_error"
        )
        neg_mean_absolute_error = scores.mean()
    except ValueError as e:
        logger.error(f"Cross-validation failed: {e}")
        raise

    metrics = {
        "train_mean_absolute_error": train_mae,
        "train_r2_score": train_r2,
        "test_mean_absolute_error": test_mae,
        "test_r2_score": test_r2,
        "neg_mean_absolute_error": neg_mean_absolute_error
    }

    logger.info("Model evaluated successfully and all the metrics are calculated.")
    return metrics

# save metrics in reports:
def save_metrics(metrics: dict, url: str) -> None:
    logger.info("Saving performance metrics ...")

    try:
        os.makedirs(os.path.dirname(url), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create reports directory: {e}")
        raise

    try:
        with open(url, "w") as file:
            json.dump(metrics, file, indent=4)
    except IOError as e:
        logger.error(f"Failed to save metrics to {url}: {e}")
        raise

    logger.info("Evaluated metrics stored successfully.")

def main() -> None:
    logger.info("Model evaluation stage start ...")

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
        logger.error(f"[FILE ERROR] {e}")
        raise

    except KeyError as e:
        logger.error(f"[SCHEMA ERROR] {e}")
        raise

    except ValueError as e:
        logger.error(f"[DATA / CONFIG ERROR] {e}")
        raise

    except RuntimeError:
        logger.error("Some runtime error occurred.")
        raise  # preserve original traceback

    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] {e}")
        raise

    logger.info("Model evaluated successfully.")


if __name__ == "__main__":
    main()