from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load("notebooks/random_forest_model_2024-09-24-16-42-34.pkl")

# Create FastAPI app
app = FastAPI()

# Define input data model
class SalesData(BaseModel):
    StoreType_a: float
    StoreType_b: float 
    StoreType_c: float
    StoreType_d: float
    Assortment_a: float 
    Assortment_b: float 
    Assortment_c: float
    PromoInterval_Feb_May_Aug_Nov: float
    PromoInterval_Jan_Apr_Jul_Oct: float
    PromoInterval_Mar_Jun_Sept_Dec: float 
    PromoInterval_None: float
    Store: float
    DayOfWeek: float
    Open: float
    Promo: float
    StateHoliday: float
    SchoolHoliday: float
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: float
    Promo2SinceWeek: float
    Promo2SinceYear: float
    Year: float
    Month: float
    Day: float
    WeekOfYear: float
    IsWeekend: float

# API endpoint for making predictions
@app.post("/predict")
async def predict_sales(data: SalesData):
    # Define the correct feature order
    feature_names = ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d',
                     'Assortment_a', 'Assortment_b', 'Assortment_c', 
                     'PromoInterval_Feb_May_Aug_Nov', 'PromoInterval_Jan_Apr_Jul_Oct',
                     'PromoInterval_Mar_Jun_Sept_Dec', 'PromoInterval_None', 
                     'Store', 'DayOfWeek', 'Open', 'Promo', 
                     'StateHoliday', 'SchoolHoliday', 'CompetitionDistance', 
                     'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 
                     'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 
                     'Month', 'Day', 'WeekOfYear', 'IsWeekend']
    
    # Convert the input data to a Pandas DataFrame to ensure feature names are preserved
    input_data = pd.DataFrame([[data.StoreType_a, data.StoreType_b, data.StoreType_c, data.StoreType_d,
                                data.Assortment_a, data.Assortment_b, data.Assortment_c, 
                                data.PromoInterval_Feb_May_Aug_Nov, data.PromoInterval_Jan_Apr_Jul_Oct,
                                data.PromoInterval_Mar_Jun_Sept_Dec, data.PromoInterval_None, 
                                data.Store, data.DayOfWeek, data.Open, data.Promo, 
                                data.StateHoliday, data.SchoolHoliday, data.CompetitionDistance, 
                                data.CompetitionOpenSinceMonth, data.CompetitionOpenSinceYear, data.Promo2, 
                                data.Promo2SinceWeek, data.Promo2SinceYear, data.Year, 
                                data.Month, data.Day, data.WeekOfYear, data.IsWeekend]],
                              columns=feature_names)

    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    
    return {"prediction": prediction[0]}
