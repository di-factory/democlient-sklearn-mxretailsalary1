# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("ml_profiling_best_API")

# Create input/output pydantic models
input_fields={
    'state': (str, ...),
    'income_employee_day': (float, ...),
    'employees_business': (int, ...),
}

output_fields={
    'prediction': (float, ...),
}

input_model = create_model("ml_profiling_best_API_input", **input_fields)
input_model2 = input_model(state= 'Veracruz', income_employee_day= 1614.612060546875, employees_business= 5)
output_model = create_model("ml_profiling_best_API_output", **output_fields)
output_model2 = output_model(prediction= 275.1473)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data:input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
