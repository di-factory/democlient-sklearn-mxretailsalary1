# API on Flask of the ML Model
# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
from flasgger import Swagger
import joblib
from src.experiment_model import MxRetailSalary1

# run app
app = Flask(__name__)
Swagger(app)


def load_pred():  # Local function to retrieve prediction model for this API
    experiment = None
    try:
        experiment = joblib.load("APIs/API_Flask/apiflask_predict.pkl")
        print("API flask predict model loaded successfully!")
    except Exception as e:
        print(f"Error loading the predict model: run first save_apiflask_model.py: {e}")

    return experiment


@app.route("/any", methods=["GET"])
async def welcome():
    """
    Welcome message
    ---
    responses:
      200:
        description: Welcome message
    """
    return {"mess": "welcome"}


# Define predict function
@app.route("/predict", methods=["POST"])
async def predict():
    """
    Endpoint to predict the target variable using the input data
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: MxRetailSalary1.Features
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
    data: MxRetailSalary1.Features = request.json
    input_data = MxRetailSalary1.Features(**data)  # validate and parse
    data = pd.DataFrame(
        [input_data.dict()], columns=experiment.feature_list
    )  # convert validated data
    predictions = experiment.predict(data)
    return {"prediction": predictions[0]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
