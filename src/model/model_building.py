import pandas as pd
import yaml
from lightgbm import LGBMRegressor
import os
import pickle

# Load data:
def load_data(train_url: str) -> tuple:
    try:
        train_df = pd.read_csv(train_url)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data not found: {train_url}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse training CSV: {e}")

    if "time" not in train_df.columns:
        raise KeyError("Target column 'time' not found in training data")

    X_train = train_df.drop(columns=["time"])
    y_train = train_df["time"]

    if X_train.empty or y_train.empty:
        raise ValueError("Training data is empty after split")

    return X_train, y_train

# load parameters:
def load_params(url: str) -> dict:
    try:
        with open(url, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {url}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")

    try:
        return params["model_building"]
    except (TypeError, KeyError):
        raise KeyError("Missing 'model_building' section in params.yaml")

# train a lightgbm model with the passed parameters:
def build_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                params: dict) -> LGBMRegressor:

    try:
        model = LGBMRegressor(**params)
    except TypeError as e:
        raise ValueError(f"Invalid LightGBM parameters: {e}")

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")

    return model

# save model:
def save_model(model: LGBMRegressor, url: str) -> None:

    path = os.path.join("models", url)

    try:
        with open(path, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        raise IOError(f"Failed to save model to {path}: {e}")


def main() -> None:
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
        raise RuntimeError(f"[FILE ERROR] {e}")

    except KeyError as e:
        raise RuntimeError(f"[SCHEMA / PARAM ERROR] {e}")

    except ValueError as e:
        raise RuntimeError(f"[CONFIG / DATA ERROR] {e}")

    except RuntimeError:
        raise  # preserve original traceback

    except Exception as e:
        raise RuntimeError(f"[UNEXPECTED ERROR] {e}")


if __name__ == "__main__":
    main()


