import pandas as pd
from api_service.preprocessing import clean_data
import joblib
import os

pd.set_option("display.max_columns", None)

data_point = pd.read_csv("api_service/query_point.csv")
data_point.drop(columns=["Unnamed: 0"], inplace=True)

# imputation and feature engineering:
trf_pipe_path = os.path.join("models", "transformer.pkl")

with open(trf_pipe_path, mode="rb") as file:
    transformer = joblib.load(file)


print(transformer.transform(data_point))