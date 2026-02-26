from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import  HTMLResponse
import os
import pandas as pd
from api_service.preprocessing import clean_data
import joblib
import logging
import mlflow
import dagshub
import pickle
import numpy as np
from fastapi.staticfiles import StaticFiles
from pathlib import Path


logger = logging.getLogger("api_service")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

app = FastAPI(debug=True)

# Serve static files (CSS, JS, images)
path  = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=path / "static"), name="static")

template_url = os.path.join(path, "templates")
templates = Jinja2Templates(directory=template_url)


# Authenticate with dagshub:
def load_model():

    logger.info("Authenticating with Dagshub ...")

    try:
        # load model from model registry:
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "PriyanshuMewal"
        repo_name = 'delivery-time-prediction'

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    except Exception as e:
        logger.error(f"Authentication Failed: {e}")
        raise

    logger.info("Dagshub Authentication Completed Successfully!")

    # loading model from mlflow model registry
    logger.info("Loading model from mlflow model registry ...")

    model_name = "LGBRegressor"
    alias = "champion"
    model_uri = f"models:/{model_name}@{alias}"

    logger.info("Model loaded successfully!")

    return mlflow.pyfunc.load_model(model_uri)

model = load_model()

def scale_output(output: np.array) -> float:

    scaler_path = os.path.join("models", "scaler.pkl")

    try:
        with open(scaler_path, mode="rb") as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        logger.error(f"Scaler file not found: {scaler_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise

    logger.info("scaler Pipeline loaded successfully.")

    try:
        new_output = scaler.inverse_transform(output.reshape(-1, 1))
    except Exception as e:
        logger.error(f"Scaling output failed because of the following error: \n {e}")
        raise

    logger.info("Scaling output Successfully ...")
    return new_output[0][0]

def feature_engineering(cleaned_query: pd.DataFrame,
                        path: str) -> pd.DataFrame:
    logger.info(f"Loading transformer pipeline from {path}.")

    trf_pipe_path = os.path.join("models", path)

    try:
        with open(trf_pipe_path, mode="rb") as file:
            transformer = joblib.load(file)
    except FileNotFoundError:
        logger.error(f"Transformer pipeline file not found: {trf_pipe_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("Transformer Pipeline loaded successfully.")

    try:
        trf_query = transformer.transform(cleaned_query)
    except Exception as e:
        logger.error(f"Feature engineering failed because of the following error: \n {e}")
        raise

    logger.info("Query Transformed Successfully ...")
    return trf_query

# Setting up fastapi home route:
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse("index.html",
                                      {"request": request,
                                       "time": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_time(request: Request,
                       driver_age: str = Form(None, description="Age of the delivery person"),
                       driver_rating: str = Form(None, description="Avg rating of the delivery person"),
                       rest_lat: float = Form(..., description="Latitude of the restaurent."),
                       rest_long: float = Form(..., description="Longitude of the restaurent."),
                       delivery_lat: float = Form(..., description="Latitude of the destination."),
                       delivery_long: float = Form(..., description="Longitude of the destination."),
                       weather: str = Form(None, description="Weather on the order date."),
                       traffic: str = Form(None, description="Traffic Level on the order date."),
                       vehicle_condition: int = Form(..., description="Vechicle condition of the delivery vehicle."),
                       vehicle_type: str = Form(..., description="Vehicle of the rider."),
                       order_type: str = Form(..., description="Type of food customer ordered.",),
                       multi_deliveries: str = Form(None, description="Number of deliveries the rider carrying"),
                       festival: str = Form(None, description="If there is a festival or not."),
                       city_type: str = Form(None, description="Type of the city."),
                       date: str = Form(..., description="The date when order is placed."),
                       order_time: str = Form(..., description="At what time the order was placed."),
                       picked_time: str = Form(..., description="At what time the order was picked by the rider.")
                       ):

    query_dict = {"age": driver_age, "ratings": driver_rating,
                 "rest_lat": rest_lat, "rest_long": rest_long,
                 "delivery_lat": delivery_lat, "delivery_long": delivery_long,
                 "date": date, "ordered_time": order_time, "picked_time": picked_time,
                 "weather": weather,"traffic": traffic,
                 "vehicle_condition": vehicle_condition, "order_type": order_type,
                 "vehicle_type": vehicle_type, "multi_deliveries": multi_deliveries,
                 "festival": festival,"city_type": city_type}

    query_point = pd.DataFrame([query_dict])

    # Data cleaning:
    cleaned_query = clean_data(query_point)

    # imputation and feature engineering:
    trf_query = feature_engineering(cleaned_query,
                                    path="transformer.pkl")

    # load model from model registry:
    scaled_time = model.predict(trf_query)

    time = scale_output(scaled_time)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "time": time
    })







