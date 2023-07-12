# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API/ml_profiling_best_API")

# Create input/output pydantic models
input_model = create_model("/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API/ml_profiling_best_API_input", **{'state': 'Jalisco', 'income_employee_day': 4245.9755859375, 'employees_business': 5})
output_model = create_model("/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API/ml_profiling_best_API_output", prediction=335.00455)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)