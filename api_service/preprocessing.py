import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("data_cleaning")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


# Calculate time of the day:
def time_of_day(col):

    if col.isna().all():
        logger.error("There are only NaN values investigate the data or the data_ingestion file.")
        raise ValueError("order_time_hour column contains only NaN values")

    return pd.cut(
        col,
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=["after_morning", "morning", "afternoon", "evening", "night"]
    )

# Calculate distance from the lat-long details:
def calculate_distance(df: pd.DataFrame):
    logger.info("Calculating distances for longitude and latitude values ...")

    required_cols = ["rest_lat", "rest_long", "delivery_lat", "delivery_long"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        logger.error(f"{missing} -> these columns are missing investigate the data or data_ingestion file.")
        raise KeyError(f"Missing columns for distance calculation: {missing}")

    try:
        lat1 = df["rest_lat"].astype(float)
        long1 = df["rest_long"].astype(float)
        lat2 = df["delivery_lat"].astype(float)
        long2 = df["delivery_long"].astype(float)
    except ValueError:
        logger.error("Latitude/Longitude columns must be numeric")
        raise

    long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = 6371 * c


# Calculate the time taken by the rider to pick the order up:
def calculate_pickup_time(df: pd.DataFrame):
    logger.info("Calculating pickup_time from existing columns ...")

    if "picked_time" not in df.columns or "ordered_time" not in df.columns:
        logger.error("Check data_ingestion file or data file because few columns are missing.")
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
    logger.info("Cleaning data ...")

    required_cols = [
        "weather", "ratings", "traffic", "age",
        "order_type", "vehicle_type", "multi_deliveries",
        "festival", "city_type", "date", "ordered_time",
        "picked_time",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"{missing} -> these columns are missing investigate the previous steps.")
        raise KeyError(f"Missing required columns: {missing}")

    # remove leading and trailing spaces:
    logger.debug("removing leading and trailing spaces from nearly all the columns and 'NaN' to np.nan.")
    cat_col = ["age", "ratings", "weather", "traffic", "order_type",
               "vehicle_type", "multi_deliveries","festival", "city_type"]

    for col in cat_col:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # remove NaN values:
    df.replace("", np.nan, inplace=True)

    # Individually cleaning each column:
    logger.debug("fix dtype of all the columns ...")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")

    # Fix date_time columns: ordered_time and picked_time  and date and more:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["ordered_time"] = pd.to_datetime(df["ordered_time"], format="%H:%M",
                                        errors="coerce").dt.time
    df["picked_time"] = pd.to_datetime(df["picked_time"], format="%H:%M",
                                       errors="coerce").dt.time

    logger.debug("generate new columns from date_time columns ...")
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
    logger.debug("remove some useless columns.")

    cols_to_drop = ["rest_lat", "rest_long", "delivery_lat",
                    "delivery_long", "date", "order_time_hour",
                     "ordered_time", "picked_time"]

    df.drop(columns=cols_to_drop, errors="ignore",  inplace=True)

    logger.info("Data cleaned successfully.")
    return df