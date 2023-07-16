# streamlit app for running model predictions
from hydra import compose, initialize
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1

import pandas as pd
import streamlit as st
import joblib
import os


def load_pred():
    experiment = None
    try:
        experiment = joblib.load('Apps/streamlit/streamlit_predict.pkl')
        print("streamlit predict model loaded successfully!")
    except Exception as e:
        print(f"Error loading the predict model: run first save_streamlit_model.py: {e}")
    
    return experiment



def main(experiment):    
    st.title("Salary Retail Estimator")
    html_temp = """
    <div style="background-color:lightblue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Salary Retail Estimator ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    state = st.text_input("Estado Mx")
    income = st.number_input(label="Ventas diarias estimadas", min_value= 4000, step = 100)
    employees = st.slider("Empleados en el local", min_value=4, max_value=8, step = 1, format='%d')
    
    result = 0.0
    
    if st.button("Predecir"):
        result = experiment.predict(
            pd.DataFrame([[state, income, employees]],
                         columns=['state', 'income_employee_day', 'employees_business']
                                    ))
    
    st.success(f'El salario (sin comisiones) es {result}')
    

if __name__ == '__main__':
    experiment = load_pred()
    main(experiment)
    
    
    
    