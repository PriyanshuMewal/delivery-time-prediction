import pandas as pd
import yaml
from lightgbm import LGBMRegressor
import os
import pickle
import logging

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# Load data:
def load_data(train_url: str) -> tuple:
    logger.info(f"Loading training data from {train_url}.")

    try:
        train_df = pd.read_csv(train_url)
    except FileNotFoundError:
        logger.error(f"Training data not found: {train_url}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse training CSV: {e}")
        raise

    if "time" not in train_df.columns:
        logger.error("'time' is missing check your preprocessing steps carefully.")
        raise KeyError("Target column 'time' not found in training data")

    X_train = train_df.drop(columns=["time"])
    y_train = train_df["time"]

    if X_train.empty or y_train.empty:
        logger.error("Input data is empty, investigate your preprocessing steps carefully.")
        raise ValueError("Training data is empty after split")

    logger.info("Training data loaded successfully.")
    return X_train, y_train

# load parameters:
def load_params(url: str) -> dict:
    logger.info(f"Loading parameters from {url}")

    try:
        with open(url, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Params file not found: {url}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML file: {e}")
        raise

    logger.info("Parameters loaded successfully.")

    try:
        return params["model_building"]
    except (TypeError, KeyError):
        logger.error("Missing 'model_building' section in params.yaml")
        raise

# train a lightgbm model with the passed parameters:
def build_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                params: dict) -> LGBMRegressor:
    logger.info("Building the model ...")

    try:
        model = LGBMRegressor(**params)
    except TypeError as e:
        logger.error(f"Invalid LightGBM parameters: {e}")
        raise

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    logger.info("Model created and trained successfully.")
    return model

# save model:
def save_model(model: LGBMRegressor, url: str) -> None:
    logger.info("Saving the trained model ...")

    path = os.path.join("models", url)

    try:
        with open(path, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise

    logger.info("model saved successfully.")


def main() -> None:
    logger.info("Model building stage start ...")

    try:
        # load data:
        data_url = "data/processed/train_processed.csv"
        X_train, y_train = load_data(data_url)

        # load parameters:
        params_url = "params.yaml"
        params = load_params(params_url)

        # train model:
        model = build_model(X_train, y_train, params)

        # save model:
        model_url = "lgbm_regressor.pkl"
        save_model(model, model_url)

    except FileNotFoundError as e:
        logger.error(f"[FILE ERROR] {e}")
        raise

    except KeyError as e:
        logger.error(f"[SCHEMA / PARAM ERROR] {e}")
        raise

    except ValueError as e:
        logger.error(f"[CONFIG / DATA ERROR] {e}")
        raise

    except RuntimeError:
        logger.error("Some Runtime error occurred.")
        raise  # preserve original traceback

    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] {e}")
        raise

    logger.info("Model Building Successfully Completed.")

if __name__ == "__main__":
    main()


