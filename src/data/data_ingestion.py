import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

# load data and hyperparameters:
df = pd.read_csv("data/external/India-Food-Delivery-Time-Prediction.csv")

with open("params.yaml", mode="rb") as file:
    test_size = yaml.safe_load(file)["data_ingestion"]["test_size"]


# drop useless columns:
df.drop(columns=['Unnamed: 0', 'ID'], inplace=True)

# rename the column names:
df.rename(columns={"Delivery_person_ID": "id",
    "Delivery_person_Age":"age",
    "Delivery_person_Ratings":"ratings",
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


# Train test split:
X = df.drop(columns=["time"])
y = df["time"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=42)

# train_df and test_df:
X_train["time"] = y_train
X_test["time"] = y_test


# Store the data:
path = os.path.join("data", "raw")

os.mkdir(path)

X_train.to_csv(os.path.join(path, "train_df.csv"), index=False)
X_test.to_csv(os.path.join(path, "test_df.csv"), index=False)
