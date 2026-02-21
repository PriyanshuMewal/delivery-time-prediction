import pandas as pd
import yaml
from lightgbm import LGBMRegressor
import os
import pickle

# Load data:
train_df = pd.read_csv("data/processed/train_processed.csv")

# Split data:
X_train = train_df.drop(columns=["time"])
y_train = train_df["time"]


# Build a lightgbm model:
with open("params.yaml", "rb") as file:
    params = yaml.safe_load(file)["model_building"]

model = LGBMRegressor(**params)

# Train model:
model.fit(X_train, y_train)

# save model:
path = os.path.join("models", "lgbm_regressor.pkl")

with open(path, mode="wb") as file:

    pickle.dump(model, file)
