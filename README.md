# Rossmann Pharmaceuticals Sales Prediction API
This project provides a Sales Prediction API for Rossmann Pharmaceuticals using a Random Forest model. The API is built with Flask and predicts future sales based on store data.

Features
Sales Prediction: Uses a trained machine learning model to predict sales.
REST API: Simple and easy-to-use endpoints for prediction.
Installation
1. Clone the repo: using git clone https://github.com/your-username/Rossmann_Pharmaceuticals.git
cd Rossmann_Pharmaceuticals
2. Install dependencies: using pip install -r requirements.txt
3.Run the Flask app:using python scripts/model_serving.py
Usage
Test the API locally at http://127.0.0.1:5000.

Use Postman to send POST requests to /predict with a JSON payload:
{
  "Store": 111,
  "DayOfWeek": 4,
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 0,
  "Customers": 150,
  "CompetitionDistance": 200
}
response seems like {
  "prediction": 5234.54
}

Folder Structure
Rossmann_Pharmaceuticals/
├── scripts/  # Contains Flask app and model scripts
├── data/                   # Datasets 
├── notebooks/        # EDA and model training notebooks
├── requirements.txt     # Dependencies
└── README.md   # Project documentation

