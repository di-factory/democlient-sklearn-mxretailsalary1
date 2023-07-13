# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from pydantic import BaseModel
import flasgger
from flasgger import Swagger

# run app
app = Flask(__name__)
Swagger(app)

# Load trained Pipeline
model = load_model("/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API_Flask/API_Flask_model")

# Create input/output pydantic models
class InputFields(BaseModel):
    state: str = 'Veracruz'
    income_employee_day: float = 1614.612060546875
    employees_business:int = 5

@app.route('/')
def welcome():
    return "wellcome"

# Define predict function
@app.route("/predict", methods=['POST'])
def predict():

    """
    Endpoint to predict the target variable using the input data.

    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: InputFields
          properties:
            state:
              type: string
              default: Veracruz
            income_employee_day:
              type: number
              format: float
              default: 1614.612060546875
            employees_business:
              type: integer
              format: int32
              default: 5
    responses:
      200:
        description: Prediction result
        schema:
          id: PredictionResult
          properties:
            prediction:
              type: number
              format: float
              description: Predicted value
    """

    data = request.json
    print(data)
    input_data = InputFields(**data) #validate and parse
    data = pd.DataFrame([input_data.dict()]) #convert validated data
    print(data)
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)