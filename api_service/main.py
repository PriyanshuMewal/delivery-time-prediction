from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import  HTMLResponse
import os
import pandas as pd
from api_service.preprocessing import clean_data
import pickle

app = FastAPI(debug=True)

template_url = os.path.join("api_service", "templates")
templates = Jinja2Templates(directory=template_url)

# Setting up fastapi home route:
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse("index.html",
                                      {"request": request,
                                       "time": None})

@app.post("/predict")
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
    trf_pipe_path = os.path.join("models", "transformer.pkl")

    with open(trf_pipe_path, mode="rb") as file:
      transformer = pickle.load(file)

    trf_query = transformer.transform(cleaned_query)

    print(trf_query)

    return query_dict





