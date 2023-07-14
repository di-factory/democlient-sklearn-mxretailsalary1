# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API_FastAPI/API_FastAPI_model")

# Create input/output pydantic models
class InputFields(BaseModel):
    state: str = 'Veracruz'
    income_employee_day: float = 1614.612060546875
    employees_business:int = 5


# Define predict function
@app.post("/predict")
def predict(data: InputFields):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
