import json
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle


# Load data:
train_df = pd.read_csv("data/processed/train_processed.csv")
test_df = pd.read_csv("data/processed/test_processed.csv")


# load model:
with open("models/lgbm_regressor.pkl", mode="rb") as file:
    lgbm_regressor = pickle.load(file)


# Split data:
X_train = train_df.drop(columns=["time"])
y_train = train_df["time"]

X_test = train_df.drop(columns=["time"])
y_test = train_df["time"]


# Evaluate model performance
y_train_pred = lgbm_regressor.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

y_pred = lgbm_regressor.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

scores = cross_val_score(lgbm_regressor, X_train, y_train, cv=5,
                         scoring="neg_mean_absolute_error")
neg_mean_absolute_error = scores.mean()


# Save all the metric values:
metrics = {
    "train_mean_absolute_error": train_mae,
    "train_r2_score": train_r2,
    "test_mean_absolute_error": test_mae,
    "test_r2_score": test_r2,
    "neg_mean_absolute_error": neg_mean_absolute_error
}

with open("reports/metrics.json", mode="w") as file:
    json.dump(metrics, file, indent=4)