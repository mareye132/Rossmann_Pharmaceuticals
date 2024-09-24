# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sales_prediction import SalesPrediction
import pandas as pd

app = Flask(__name__)

# Load the trained model (from Task 2)
model = joblib.load("random_forest_model_2024-09-23.pkl")  # Update with actual model filename

# Preprocessing for the input data
def preprocess_input(data):
    # Add your preprocessing steps here based on how the model was trained
    # This might involve scaling, encoding, etc.
    # Ensure that the input format matches what the model expects
    df = pd.DataFrame([data])
    return df

@app.route('/')
def home():
    return "Sales Prediction API is up and running!"

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json

        # Preprocess the input data (customize this as needed)
        processed_data = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Send response
        response = {
            'prediction': prediction[0]  # Assuming model predicts a single value (sales)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
