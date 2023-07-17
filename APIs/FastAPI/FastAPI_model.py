# API on Flask of the ML Model
# -*- coding: utf-8 -*

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from src.experiment_model import MxRetailSalary1
import joblib

# Create the app
app = FastAPI()

# Create input/output pydantic models
class InputFields(BaseModel):
    state: str = 'Veracruz'
    income_employee_day: float = 1614.612060546875
    employees_business: int = 5


def load_pred():  # Local function to retrieve prediction model for this API
    experiment = None
    try:
        experiment = joblib.load('APIs/FastAPI/fastapi_predict.pkl')
        print("FastAPI predict model loaded successfully!")
    except Exception as e:
        print(f"Error loading the predict model: run first save_fastapi_model.py: {e}")
    
    return experiment


# Define predict function

@app.get('/')
def welcome():
    return "wellcome"


@app.post("/predict")
def predict(data: InputFields):
    experiment = load_pred()

    input = pd.DataFrame([data.dict()], columns=['state', 'income_employee_day', 'employees_business'])
    predictions = experiment.predict(input)  
    return {"prediction": predictions[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
