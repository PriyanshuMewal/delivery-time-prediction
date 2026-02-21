import pandas as pd
import numpy as np
import os

# load data:
train_df = pd.read_csv("data/raw/train_df.csv")
test_df = pd.read_csv("data/raw/test_df.csv")


# Calculate time of the day:
def time_of_day(col):

    return pd.cut(col, bins=[0,6,12,17,20,24], right=True,
                 labels=["after_morning", "morning", "afternoon", "evening", "night"])


# Calculate distance from the lat-long details:
def calculate_distance(df):
    lat1 = df["rest_lat"]
    long1 = df["rest_long"]
    lat2 = df["delivery_lat"]
    long2 = df["delivery_long"]

    long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    distance = 6371 * c

    df["distance"] = distance


# Calculate the time taken by the rider to pick the order up:
def calculate_pickup_time(df):

    picked_time = pd.to_datetime(df["picked_time"].astype(str),
                                 format="%H:%M:%S")
    ordered_time = pd.to_datetime(df["ordered_time"].astype(str),
                                  format="%H:%M:%S")

    return np.where(ordered_time < picked_time,
             (picked_time - ordered_time).dt.seconds / 60,
                np.NaN)


# Clean data:
def clean_data(df):

    # fix some values
    df["weather"] = df["weather"].str.replace("conditions ", "").str.strip()
    df["time"] = df["time"].str.split(")").str.get(1).str.strip().astype(np.float32)
    df["id"] = df["id"].str.split("RES").str.get(0)
    df.rename(columns={"id": "city"}, inplace=True)

    # remove leading and trailing spaces:
    cat_col = ["age", "ratings", "city", "weather", "traffic", "order_type",
               "vehicle_type", "multi_deliveries","festival", "city_type"]

    for col in cat_col:
        df[col] = df[col].str.strip().str.lower()

    # remove NaN values:
    df.replace("NaN", np.nan, inplace=True)

    # Individually cleaning each column:
    # age:
    df["age"] = df["age"].astype(np.float32)
    df.drop(index=df[df["age"] < 18].index, inplace=True)

    # ratings:
    df["ratings"] = df["ratings"].astype(np.float32)
    df.drop(index=df[df["ratings"] > 5].index, inplace=True)

    # reset_index:
    df.reset_index(drop=True, inplace=True)

    # convert date column to datetime:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

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

    df.drop(columns=cols_to_drop, inplace=True)


# clean training and testing data:
clean_data(train_df)
clean_data(test_df)


# store data:
path = os.path.join("data", "interim")

os.mkdir(path)

train_df.to_csv(os.path.join(path, "train_cleaned.csv"), index=False)
test_df.to_csv(os.path.join(path, "test_cleaned.csv"), index=False)