import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import os
import pickle
import yaml
import joblib
from src.features.utility import ModeImputation

from sklearn import set_config
set_config(transform_output="pandas")
import logging

logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


# load data and parameters:
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
        logger.error("Target column 'time' is missing ")
        raise KeyError("Target column 'time' not found in input data")

    X_train = train_df.drop(columns=["time"])
    y_train = train_df["time"]

    X_test = test_df.drop(columns=["time"])
    y_test = test_df["time"]

    logger.info("Data loaded successfully.")
    return X_train, X_test, y_train, y_test

def load_params(param_url: str) -> dict:
    logger.info(f"Loading parameter from {param_url}.")

    try:
        with open(param_url, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Params file not found: {param_url}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML file: {e}")
        raise

    logger.info("Parameters loaded successfully.")

    try:
        return params["feature_engineering"]
    except (TypeError, KeyError):
        logger.error("Missing 'feature_engineering' section in params.yaml")
        raise


# Create pipeline to transform the features:
def create_transformer_pipeline(params: dict) -> Pipeline:
    logger.info("Creating transformation pipeline ...")

    required_keys = [
        "const_imputation",
        "iterative_imputation",
        "ordinal_encoder",
        "onehot_encoder",
        "lgbm_estimator"
    ]

    missing = [k for k in required_keys if k not in params]
    if missing:
        logger.error(f"{missing} -> these params are missing check the params.yaml")
        raise KeyError(f"Missing required parameters: {missing}")

    # ColumnTransformers for Imputation:
    logger.debug("Creating imputation column-transformers ...")

    num_cols = ["age", "ratings", "pickup_time"]
    mode_impute = ["multi_deliveries", "festival", "city_type"]
    random_impute = ["weather", "traffic", "order_time_of_day"]

    # parameters for imputation:
    strategy = params["const_imputation"]["strategy"]
    fill_value = params["const_imputation"]["fill_value"]


    impute_categorical_const = ColumnTransformer(transformers=[
        ("const_imputation", SimpleImputer(strategy=strategy,
                                           fill_value=fill_value), random_impute),
    ], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)

    impute_categorical_const.set_output(transform="pandas")


    lgbm_params = params["lgbm_estimator"]
    lgb_estimator = LGBMRegressor(**lgbm_params)
    max_iter = params["iterative_imputation"]["max_iter"]

    impute_numerical_iterative = ColumnTransformer(transformers=[
       ("iterative", IterativeImputer(estimator=lgb_estimator,
                                      max_iter=max_iter), num_cols)
    ], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)

    impute_numerical_iterative.set_output(transform="pandas")


    # ColumnTransformers for Feature Engineering:
    logger.debug("Creating column-transformer for encoding categorical columns.")

    nom_cat = ["weather", "order_type", "vehicle_type", "festival", "city_type"]
    ord_cat = ["traffic", "distance_type", "order_time_of_day"]
    numerical = ["age", "ratings", "pickup_time", "distance"]

    traffic_categories = ['low', 'medium', 'high', 'jam']
    distance_type_categories = ['short', 'medium', 'long', 'very_long']
    time_categories = ['morning', 'afternoon', "evening", 'night']


    # parameter for transformation:
    unknown_value = params["ordinal_encoder"]["unknown_value"]
    ohe_params = params["onehot_encoder"]

    trf_categorical = ColumnTransformer(transformers=[
        ("ord_cat", OrdinalEncoder(categories=[traffic_categories, distance_type_categories, time_categories],
                                  handle_unknown="use_encoded_value",
                                   unknown_value=unknown_value), ord_cat),
        ("nom_cat", OneHotEncoder(**ohe_params), nom_cat)
    ], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)

    trf_categorical.set_output(transform="pandas")

    trf_numerical = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), numerical)
    ], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)

    trf_numerical.set_output(transform="pandas")


    logger.debug("Creating final pipeline ...")
    final_preprocessing = Pipeline(steps=[
        ("mode_impue", ModeImputation(mode_impute)),
        ("impute_cat", impute_categorical_const),
        ("trf_cat", trf_categorical),
        ("impute_num", impute_numerical_iterative),
        ("trf_num", trf_numerical),
    ])

    logger.info("Pipeline created.")
    return final_preprocessing


# Transform feature space and target column:
def transform_features(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       pipeline: Pipeline):
    logger.info("transforming features -> imputation + ecoding and scaling")

    try:
        processed_X_train = pipeline.fit_transform(X_train)
        processed_X_test = pipeline.transform(X_test)
    except Exception as e:
        logger.error(f"Feature transformation failed: {e}")
        raise

    logger.info("features transformed successfully.")
    return processed_X_train, processed_X_test

def transform_target(y_train: pd.Series,
                     y_test: pd.Series):
    logger.info("Transform target column ...")

    if y_train.empty or y_test.empty:
        logger.error("Oh!, wait it's empty bro, check the preprocessing pipeline.")
        raise ValueError("Target series is empty")

    try:
        scaler = StandardScaler()
        y_train_scaled = pd.Series(
            scaler.fit_transform(y_train.values.reshape(-1, 1)).values.ravel(),
            index=y_train.index,
            name=y_train.name
        )
        y_test_scaled = pd.Series(
            scaler.transform(y_test.values.reshape(-1, 1)).values.ravel(),
            index=y_test.index,
            name=y_test.name
        )
    except Exception as e:
        logger.error(f"Target scaling failed: {e}")
        raise

    logger.info("Target scaled successfully.")
    return y_train_scaled, y_test_scaled, scaler

# Store transformed data and Standard Scaler:
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
              data_url: str, scaler: StandardScaler) -> None:
    logger.info("Saving the training and testing data after feature engineering and also the scaler for target")

    path = os.path.join("data", data_url)

    try:
        os.makedirs(path, exist_ok=True)
        os.makedirs("models", exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directories: {e}")
        raise

    try:
        train_df.to_csv(os.path.join(path, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(path, "test_processed.csv"), index=False)
    except IOError as e:
        logger.error(f"Failed to save processed CSV files: {e}")
        raise

    try:
        with open(os.path.join("models", "scaler.pkl"), "wb") as file:
            pickle.dump(scaler, file)
    except IOError as e:
        logger.error(f"Failed to save scaler object: {e}")
        raise

    logger.info("Everything saved successfully!")

# Save the transformer pipeline:
def save_pipeline(pipeline: Pipeline, file_name: str) -> None:
    logger.info("Saving the pipeline after feature engineering.")

    try:
        os.makedirs("models", exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    pipe_url = os.path.join("models", file_name)

    try:
        joblib.dump(pipeline, pipe_url)
    except IOError as e:
        logger.error(f"Failed to save the pipeline object: {e}")
        raise

    logger.info("pipeline saved successfully!")


def main() -> None:
    logger.info("feature engineering stage started ...")
    try:
        train_url = "data/interim/train_cleaned.csv"
        test_url = "data/interim/test_cleaned.csv"

        X_train, X_test, y_train, y_test = load_data(train_url, test_url)

        param_url = "params.yaml"
        params = load_params(param_url)

        pipeline = create_transformer_pipeline(params)

        X_train_trf, X_test_trf = transform_features(X_train, X_test, pipeline)
        y_train_trf, y_test_trf, scaler = transform_target(y_train, y_test)

        X_train_trf["time"] = y_train_trf
        X_test_trf["time"] = y_test_trf

        pipeline_url = "transformer.pkl"
        save_pipeline(pipeline, pipeline_url)

        save_data(X_train_trf, X_test_trf, "processed", scaler)

    except FileNotFoundError as e:
        logger.error(f"[FILE ERROR] {e}")
        raise

    except KeyError as e:
        logger.error(f"[SCHEMA / PARAM ERROR] {e}")
        raise

    except ValueError as e:
        logger.error(f"[DATA ERROR] {e}")
        raise

    except RuntimeError:
        logger.error("Some runtime error occurred.")
        raise

    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] {e}")
        raise

    logger.info("Feature engineering stage completed successfully.")


if __name__ == "__main__":
    main()
