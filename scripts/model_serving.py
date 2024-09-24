from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load("C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/random_forest_model_2024-09-22-13-13-10.pkl")

# Preprocessing for the input data
def preprocess_input(data):
    # Convert the 'Date' feature into useful features like DayOfYear
    date_str = data['Date']  # Expecting 'YYYY-MM-DD' format
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday  # Day of the year
    
    # Map 'StateHoliday' to numerical values or one-hot encode if necessary
    state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}  # Example map for 'StateHoliday'
    data['StateHoliday'] = state_holiday_map.get(data['StateHoliday'], 0)  # Default to 0 if not found

    # Add any other features that your model expects, e.g., StoreType, Assortment, etc.
    # Assuming these features are present in your input or you need to derive them.
    # Example placeholder for other features
    features = [
        data['Store'], 
        data['DayOfWeek'], 
        day_of_year,  # You can include the original 'Date' if necessary
        data['Customers'], 
        data['Open'], 
        data['Promo'], 
        data['StateHoliday'], 
        data['SchoolHoliday'],
        # Add more features based on the original training data (e.g., StoreType, Assortment, etc.)
        data.get('StoreType', 0),  # Placeholder: 'StoreType' might need encoding (or one-hot encoding)
        data.get('Assortment', 0),  # Placeholder: 'Assortment'
        data.get('CompetitionDistance', 0),  # Placeholder: 'CompetitionDistance'
        # Add all other required features
    ]
    
    # Convert to DataFrame for easier manipulation (if necessary)
    df = pd.DataFrame([features], columns=[
        'Store', 'DayOfWeek', 'DayOfYear', 'Customers', 'Open', 'Promo', 
        'StateHoliday', 'SchoolHoliday',
        # Add all other feature names here (StoreType, Assortment, CompetitionDistance, etc.)
    ])

    # Ensure all 25 features are passed to the model as expected
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

        # Preprocess the input data
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
