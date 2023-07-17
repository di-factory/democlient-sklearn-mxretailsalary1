# API on Flask of the ML Model
# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
from pydantic import BaseModel
from flasgger import Swagger
import joblib

# run app
app = Flask(__name__)
Swagger(app)


def load_pred():  # Local function to retrieve prediction model for this API
    experiment = None
    try:
        experiment = joblib.load('APIs/API_Flask/apiflask_predict.pkl')
        print("API flask predict model loaded successfully!")
    except Exception as e:
        print(f"Error loading the predict model: run first save_apiflask_model.py: {e}")
    
    return experiment


# Create input/output pydantic models
class InputFields(BaseModel):
    state: str = 'Veracruz'
    income_employee_day: float = 1614.612060546875
    employees_business: int = 5


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

    experiment = load_pred()
    data = request.json
    input_data = InputFields(**data)  # validate and parse
    data = pd.DataFrame([input_data.dict()], columns=['state', 'income_employee_day', 'employees_business'])  # convert validated data
    predictions = experiment.predict(data)
    return {"prediction": predictions[0]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)