import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

# load data:
def load_data(path: str) -> tuple:

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at path: {path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    # drop useless columns:
    df.drop(columns=['Unnamed: 0', 'ID'], errors="ignore", inplace=True)

    # rename the column names:
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
        raise KeyError("Target column 'time' not found after renaming")

    # Train test split:
    X = df.drop(columns=["time"])
    y = df["time"]

    return X, y

# load hyperparameters:
def load_params(path: str) -> float:

    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Params file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")

    try:
        test_size = params["data_ingestion"]["test_size"]
    except (TypeError, KeyError):
        raise KeyError("Missing 'data_ingestion.test_size' in params.yaml")

    if not isinstance(test_size, float):
        raise TypeError("test_size must be a float")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    return test_size

# Store the data:
def save_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame, url: str) -> None:
    path = os.path.join("data", url)

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}")

    try:
        train_df.to_csv(os.path.join(path, "train_df.csv"), index=False)
        test_df.to_csv(os.path.join(path, "test_df.csv"), index=False)
    except IOError as e:
        raise IOError(f"Failed to save CSV files: {e}")


def main() -> None:

    data_url = "data/external/India-Food-Delivery-Time-Prediction.csv"
    param_url = "params.yaml"

    X, y = load_data(data_url)
    test_size = load_params(param_url)

    # Split the data:
    if X.empty or y.empty:
        raise ValueError("Loaded dataset is empty")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )
    except ValueError as e:
        raise ValueError(f"Train-test split failed: {e}")

    # train_df and test_df:
    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["time"] = y_train.values
    test_df["time"] = y_test.values

    # save data:
    folder = "raw"
    save_data(train_df, test_df, folder)


if __name__ == "__main__":
    main()

