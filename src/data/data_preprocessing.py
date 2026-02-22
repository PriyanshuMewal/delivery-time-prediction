import pandas as pd
import numpy as np
import os


# load data:
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

    return train_df, test_df


# Calculate time of the day:
def time_of_day(col):
    if col.isna().all():
        raise ValueError("order_time_hour column contains only NaN values")

    return pd.cut(
        col,
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=["after_morning", "morning", "afternoon", "evening", "night"]
    )

# Calculate distance from the lat-long details:
def calculate_distance(df: pd.DataFrame):

    required_cols = ["rest_lat", "rest_long", "delivery_lat", "delivery_long"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for distance calculation: {missing}")

    try:
        lat1 = df["rest_lat"].astype(float)
        long1 = df["rest_long"].astype(float)
        lat2 = df["delivery_lat"].astype(float)
        long2 = df["delivery_long"].astype(float)
    except ValueError:
        raise ValueError("Latitude/Longitude columns must be numeric")

    long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = 6371 * c


# Calculate the time taken by the rider to pick the order up:
def calculate_pickup_time(df: pd.DataFrame):

    if "picked_time" not in df.columns or "ordered_time" not in df.columns:
        raise KeyError("Missing picked_time or ordered_time columns")

    picked_time = pd.to_datetime(
        df["picked_time"].astype(str),
        format="%H:%M:%S",
        errors="coerce"
    )

    ordered_time = pd.to_datetime(
        df["ordered_time"].astype(str),
        format="%H:%M:%S",
        errors="coerce"
    )

    return np.where(
        ordered_time < picked_time,
        (picked_time - ordered_time).dt.seconds / 60,
        np.nan
    )

# Clean data:
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    required_cols = [
        "weather", "time", "id", "age", "ratings", "traffic",
        "order_type", "vehicle_type", "multi_deliveries",
        "festival", "city_type", "date", "ordered_time",
        "picked_time"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # fix string columns safely
    df["weather"] = df["weather"].astype(str).str.replace("conditions ", "").str.strip()

    # fix some values
    try:
        df["time"] = (
            df["time"]
            .astype(str)
            .str.split(")")
            .str.get(1)
            .str.strip()
            .astype(np.float32)
        )
    except ValueError:
        raise ValueError("Failed to parse target column 'time'")

    df["id"] = df["id"].astype(str).str.split("RES").str.get(0)
    df.rename(columns={"id": "city"}, inplace=True)

    # remove leading and trailing spaces:
    cat_col = ["age", "ratings", "city", "weather", "traffic", "order_type",
               "vehicle_type", "multi_deliveries","festival", "city_type"]

    for col in cat_col:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # remove NaN values:
    df.replace("NaN", np.nan, inplace=True)

    # Individually cleaning each column:
    # age:
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.drop(index=df[df["age"] < 18].index, inplace=True)

    # ratings:
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")
    df.drop(index=df[df["ratings"] > 5].index, inplace=True)

    # reset_index:
    df.reset_index(drop=True, inplace=True)

    # convert date column to datetime:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

    # Change ordered_time and picked_time to time object:
    df["ordered_time"] = pd.to_datetime(df["ordered_time"], format="%H:%M:%S",
                                        errors="coerce").dt.time
    df["picked_time"] = pd.to_datetime(df["picked_time"], format="%H:%M:%S",
                                       errors="coerce").dt.time

    df["order_day"] = df["date"].dt.day
    df["order_month"] = df["date"].dt.month
    df["order_day_of_week"] = df["date"].dt.day_name().str.lower()
    df["is_weekend"] = df["date"].dt.day_name().isin(["Sunday", "Saturday"]).astype(int)
    df["order_time_hour"] = pd.to_datetime(df["ordered_time"], format="%H:%M:%S",
                                           errors="coerce").dt.hour
    df["order_time_of_day"] = time_of_day(df["order_time_hour"])
    df["pickup_time"] = calculate_pickup_time(df)

    # Distance and distance_type:
    calculate_distance(df)
    df["distance_type"] = pd.cut(df["distance"], [0, 5, 10, 15, 25],
                                 right=False, labels=["short", "medium",
                                                      "long", "very_long"])

    # drop rows where missing values are more than 7:
    df = df[df.isna().sum(axis=1) <= 7]

    # Drop unnecessary columns:
    cols_to_drop = ["rest_lat", "rest_long", "delivery_lat",
                    "delivery_long", "date", "order_time_hour",
                    "order_day", "city", "order_day_of_week",
                    "ordered_time", "picked_time"]

    df.drop(columns=cols_to_drop, errors="ignore",  inplace=True)
    
    return df


# store data:
def save_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame, url: str) -> None:

    path = os.path.join("data", url)

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}")

    try:
        train_df.to_csv(os.path.join(path, "train_cleaned.csv"), index=False)
        test_df.to_csv(os.path.join(path, "test_cleaned.csv"), index=False)
    except IOError as e:
        raise IOError(f"Failed to save cleaned CSV files: {e}")


def main() -> None:
    try:
        # load data:
        train_url = "data/raw/train_df.csv"
        test_url = "data/raw/test_df.csv"

        train_df, test_df = load_data(train_url, test_url)

        # clean data:
        train_cleaned = clean_data(train_df)
        test_cleaned = clean_data(test_df)

        # save data:
        url = "interim"
        save_data(train_cleaned, test_cleaned, url)

    except FileNotFoundError as e:
        raise RuntimeError(f"[DATA LOADING ERROR] {e}")

    except KeyError as e:
        raise RuntimeError(f"[SCHEMA ERROR] Missing or invalid column: {e}")

    except ValueError as e:
        raise RuntimeError(f"[DATA VALIDATION ERROR] {e}")

    except OSError as e:
        raise RuntimeError(f"[FILE SYSTEM ERROR] {e}")


if __name__ == "__main__":
    main()