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
from sklearn import set_config
set_config(transform_output="pandas")

# load data and parameters:
train_df = pd.read_csv("data/interim/train_cleaned.csv")
test_df = pd.read_csv("data/interim/test_cleaned.csv")

with open("params.yaml", mode="rb") as file:
    params = yaml.safe_load(file)["feature_engineering"]

X_train = train_df.drop(columns=["time"])
y_train = train_df["time"]

X_test = test_df.drop(columns=["time"])
y_test = test_df["time"]


# Mode imputation:
def mode_imputation(X):

    modes_ = X.mode(dropna=True).iloc[0]
    dtypes_ = X.dtypes

    for col in X.columns:
        X[col] = X[col].fillna(modes_[col]).astype(dtypes_[col])

    return X

# Create column_transformer for further transformations:

# ColumnTransformers for Imputation:
num_cols = ["age", "ratings", "pickup_time"]
mode_impute = ["multi_deliveries", "festival", "city_type"]
random_impute = ["weather", "traffic", "order_time_of_day"]

# parameters for imputation:
strategy = params["const_imputation"]["strategy"]
fill_value = params["const_imputation"]["fill_value"]

impute_categorical_const = ColumnTransformer(transformers=[
    ("mode_imputation", FunctionTransformer(mode_imputation), mode_impute),
    ("const_imputation", SimpleImputer(strategy=strategy,
                                       fill_value=fill_value), random_impute),
], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)


lgbm_params = params["lightgbm_params"]
lgb_estimator = LGBMRegressor(**lgbm_params)
max_iter = params["iterative_imputation"]["max_iter"]

impute_numerical_iterative = ColumnTransformer(transformers=[
   ("iterative", IterativeImputer(estimator=lgb_estimator,
                                  max_iter=max_iter), num_cols)
], remainder="passthrough", n_jobs=-1, verbose_feature_names_out=False)

# ColumnTransformers for Feature Engineering:
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

final_preprocessing = Pipeline(steps=[
    ("impute_cat", impute_categorical_const),
    ("trf_cat", trf_categorical),
    ("impute_num", impute_numerical_iterative),
    ("trf_num", trf_numerical),
])


# Transform feature space:
processed_X_train = final_preprocessing.fit_transform(X_train)
processed_X_test = final_preprocessing.transform(X_test)

# Transform Target Column:
scaler = StandardScaler()

y_train_scaled = pd.Series(scaler.fit_transform(y_train.values.reshape(y_train.shape[0], 1)).values.ravel(),
                           index=y_train.index, name=y_train.name)
y_test_scaled = pd.Series(scaler.transform(y_test.values.reshape(y_test.shape[0], 1)).values.ravel(),
                           index=y_test.index, name=y_test.name)

processed_X_train["time"] = y_train_scaled
processed_X_test["time"] = y_test_scaled


# Store transformed data and Standard Scaler:
path = os.path.join("data", "processed")

os.mkdir(path)

processed_X_train.to_csv(os.path.join(path, "train_processed.csv"), index=False)
processed_X_test.to_csv(os.path.join(path, "test_processed.csv"),  index=False)

with open(os.path.join("reports", "scaler.pkl"), "wb") as file:
    pickle.dump(scaler, file)

