from flask import Flask, render_template, request
import pandas as pd
import joblib

# Run app
app = Flask(__name__)

def load_pred():  # Local function to retrieve prediction model for this Streamlit App
    experiment = None
    try:
        experiment = joblib.load('Apps/Flask/flask_predict.pkl')
        print("streamlit predict model loaded successfully!")
    except Exception as e:
        print(f"Error loading the predict model: run first save_streamlit_model.py: {e}")
    
    return experiment

@app.route('/', methods=['GET', 'POST'])
def main():
    experiment = load_pred()
    result = 0.0
    
    if request.method == 'POST':
        state = request.form['state']
        income = float(request.form['income'])
        employees = int(request.form['employees'])
        
        input_data = pd.DataFrame([[state, income, employees]], columns=experiment.feature_list)
        result = experiment.predict(input_data)[0]
    
    return render_template('index2.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
