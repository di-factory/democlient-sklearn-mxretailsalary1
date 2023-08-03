# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model(
    "/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/models/ml_profiling_best"
)

# Create input/output pydantic models
input_model = create_model(
    "/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/models/ml_profiling_best_input",
    **{
        "state": "Oaxaca",
        "income_employee_day": 3448.222412109375,
        "employees_business": 4,
    }
)
output_model = create_model(
    "/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/models/ml_profiling_best_output",
    prediction=395.38654,
)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
