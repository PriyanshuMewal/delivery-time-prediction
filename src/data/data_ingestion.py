import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# load data:
def load_data(path: str) -> tuple:
    logger.info(f"Loading data from {path}")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Data file not found at path: {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise

    # drop useless columns:
    logger.info("Dropping useless columns (if present).")
    df.drop(columns=['Unnamed: 0', 'ID'], errors="ignore", inplace=True)

    # rename the column names:
    logger.info("Renaming the remaining columns.")
    df.rename(columns={"Delivery_person_ID": "id",
                       "Delivery_person_Age": "age",
                       "Delivery_person_Ratings": "ratings",
                       "Restaurant_latitude": "rest_lat",
                       "Restaurant_longitude": "rest_long",
                       "Delivery_location_latitude": "delivery_lat",
                       "Delivery_location_longitude": "delivery_long",
                       "Order_Date": "date",
                       "Time_Orderd": "ordered_time",
                       "Time_Order_picked": "picked_time",
                       "Weatherconditions": "weather",
                       "Road_traffic_density": "traffic",
                       "Vehicle_condition": "vehicle_condition",
                       "Type_of_order": "order_type",
                       "Type_of_vehicle": "vehicle_type",
                       "multiple_deliveries": "multi_deliveries",
                       "Festival": "festival",
                       "City": "city_type",
                       "Time_taken(min)": "time"}, inplace=True)

    if "time" not in df.columns:
        logger.error("Target column 'time' missing after renaming")
        raise KeyError("Target column 'time' not found.")


    # Train test split:
    X = df.drop(columns=["time"])
    y = df["time"]

    logger.info("Data Loaded Successfully.")
    return X, y

# load hyperparameters:
def load_params(path: str) -> float:
    logger.info(f"Loading parameter from {path}.")

    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Params file not found: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML file: {e}")
        raise

    try:
        test_size = params["data_ingestion"]["test_size"]
    except (TypeError, KeyError):
        logger.error("Missing 'data_ingestion.test_size' in params.yaml")
        raise

    if not isinstance(test_size, float):
        logger.error("Check for the value of test_size in params.yaml.")
        raise TypeError("test_size must be a float")

    if not 0 < test_size < 1:
        logger.error("We are taking test size as the percentage of entire data to create test dataset")
        raise ValueError("test_size must be between 0 and 1")

    logger.info("Parameters loaded successfully.")
    return test_size

# Store the data:
def save_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame, url: str) -> None:
    logger.info(f"Saving training and test data to {url}")

    path = os.path.join("data", url)

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

    try:
        train_df.to_csv(os.path.join(path, "train_df.csv"), index=False)
        test_df.to_csv(os.path.join(path, "test_df.csv"), index=False)
    except IOError as e:
        logger.error(f"Failed to save CSV files: {e}")
        raise

    logger.info("Data saved successfully.")


def main() -> None:
    logger.info("Data ingestion stage started ...")

    data_url = "data/external/India-Food-Delivery-Time-Prediction.csv"
    param_url = "params.yaml"

    X, y = load_data(data_url)
    test_size = load_params(param_url)

    # Split the data:
    if X.empty or y.empty:
        logger.error("Either your feature space or target column is empty.")
        raise ValueError("Loaded dataset is empty")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )
    except ValueError as e:
        logger.error(f"Train-test split failed: {e}")
        raise

    # train_df and test_df:
    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["time"] = y_train.values
    test_df["time"] = y_test.values

    # save data:
    folder = "raw"
    save_data(train_df, test_df, folder)

    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()

