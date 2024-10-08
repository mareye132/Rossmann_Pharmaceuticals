

#Task-1: Exploratory Data Analysis (EDA)
Introduction
This section outlines the Exploratory Data Analysis (EDA) performed on the Rossmann Pharmaceuticals dataset as part of Task-1. EDA is a crucial step in data analysis that helps in understanding the underlying patterns, detecting anomalies, and checking assumptions through summary statistics and visualizations. The primary objective of this analysis is to extract insights that inform business decisions, particularly regarding sales performance and the impact of promotions.

Points Covered in Task-1:
Data Loading and Logging:

Successfully loaded the store and training datasets using Pandas.
Implemented logging functionality to track the data loading process and other key steps in the analysis.
Missing Values Analysis:

Conducted a thorough check for missing values in both datasets.
Logged the missing values for review and subsequent handling.
Addressed missing values in critical columns to maintain data integrity.
Distribution Check:

Analyzed the distribution of promotions in both training and test sets to determine if they were similarly distributed, ensuring comparability between the datasets.
Sales Behavior Analysis:

Compared sales behavior across different time periods, specifically before, during, and after holidays.
Investigated seasonal purchase behaviors, focusing on trends associated with significant holidays such as Christmas and Easter.
Correlation Analysis:

Evaluated the correlation between sales and the number of customers to understand the strength and direction of their relationship.
Promotion Impact:

Assessed the effects of promotions on sales, exploring whether they attract new customers and how they impact existing customer behavior.
Effective Promotion Deployment:

Identified opportunities for more effective promotion strategies, determining which stores could benefit from targeted promotional efforts.
Customer Behavior Trends:

Examined trends in customer behavior relative to store opening and closing times, providing insights into optimal operating hours.
Store Operations:

Identified stores that are open on all weekdays and analyzed how this consistent availability affects their sales performance on weekends.
Assortment Type Effect:

Investigated the impact of different product assortment types on sales, providing insights into consumer preferences.
Competitor Distance Analysis:

Analyzed how the distance to the nearest competitor affects sales, particularly in urban environments where multiple competitors may exist.
Competitor Impact:

Evaluated the impact of new competitors entering the market, specifically examining stores with initially unknown competitor distances that later received valid data.
Visualization of Results:
The analysis included visualizations, such as bar plots, to illustrate the impact of holidays on sales, highlighting average sales figures before, during, and after holiday periods.

# Rossmann_Pharmaceuticals
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


