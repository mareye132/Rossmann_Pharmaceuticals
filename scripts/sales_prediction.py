import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class SalesPrediction:
    def __init__(self, store_data_path, train_data_path):
        self.store_data_path = store_data_path
        self.train_data_path = train_data_path
        self.scaler = StandardScaler()
        self.model_pipeline = None
        self.lstm_model = None

    def load_data(self):
        store_df = pd.read_csv(self.store_data_path)
        train_df = pd.read_csv(self.train_data_path, low_memory=False)
        return pd.merge(train_df, store_df, how='left', on='Store')

    def preprocess_data(self, df):
        # Feature Engineering
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weekday'] = df['Date'].dt.weekday
        df['IsWeekend'] = df['Weekday'] >= 5
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfMonth'] = df['Date'].dt.day
        df['IsStartOfMonth'] = df['DayOfMonth'] <= 10
        df['IsMidOfMonth'] = (df['DayOfMonth'] > 10) & (df['DayOfMonth'] <= 20)
        df['IsEndOfMonth'] = df['DayOfMonth'] > 20
        df['DaysToNextHoliday'] = (pd.Timestamp('2024-12-25') - df['Date']).dt.days
        df['LastHoliday'] = pd.Timestamp('2024-01-01')  # Example holiday date
        df['DaysAfterLastHoliday'] = (df['Date'] - df['LastHoliday']).dt.days.clip(lower=0)

        # Convert non-numeric columns to numeric, handling errors
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Fill NaN values
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Splitting features and target
        X = df.drop(['Sales', 'Date'], axis=1)
        y = df['Sales']
        return X, y

    def check_stationarity(self, df, column='Sales'):
        result = adfuller(df[column])
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        if result[1] < 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")

    def plot_acf_pacf(self, df, column='Sales'):
        plot_acf(df[column])
        plot_pacf(df[column])
        plt.show()

    def split_and_scale_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_and_train_model(self, X_train, y_train):
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.model_pipeline.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test, loss_function='mse'):
        y_pred = self.model_pipeline.predict(X_test)
        if loss_function == 'mse':
            mse = mean_squared_error(y_test, y_pred)
            print(f"Mean Squared Error: {mse}")
        elif loss_function == 'mae':
            mae = mean_absolute_error(y_test, y_pred)
            print(f"Mean Absolute Error: {mae}")
        return y_pred

    def feature_importance(self, X_train_original):
        importances = self.model_pipeline.named_steps['random_forest'].feature_importances_
        feature_importance = pd.Series(importances, index=X_train_original.columns).sort_values(ascending=False)
        return feature_importance

    def estimate_confidence_interval(self, y_test, y_pred, confidence=0.95):
        error = np.abs(y_test - y_pred)
        margin = error.mean() + confidence * error.std()
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        print(f"Confidence Interval: [{lower_bound.mean()}, {upper_bound.mean()}]")
        return lower_bound, upper_bound

    def save_model(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        joblib.dump(self.model_pipeline, f"random_forest_model_{timestamp}.pkl")

    def create_sequences(self, data, sequence_length):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:i + sequence_length]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def prepare_lstm_data(self, df):
        df = df.sort_values('Date')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_sales = scaler.fit_transform(df[['Sales']])
        X, y = self.create_sequences(scaled_sales, sequence_length=30)
        return X, y, scaler

    def build_lstm_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        return model

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        return self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def save_lstm_model(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.lstm_model.save(f"lstm_model_{timestamp}.h5")

# Main function to execute the workflow
def main():
    store_data_path = 'C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/Data/store.csv'  # Update with actual path
    train_data_path = 'C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/Data/train.csv'   # Update with actual path

    sales_prediction = SalesPrediction(store_data_path, train_data_path)

    # Load data
    data = sales_prediction.load_data()

    # Preprocess data
    X, y = sales_prediction.preprocess_data(data)

    # Check for stationarity
    sales_prediction.check_stationarity(data)

    # Plot ACF and PACF
    sales_prediction.plot_acf_pacf(data)

    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test = sales_prediction.split_and_scale_data(X, y)

    # Build and train model
    sales_prediction.build_and_train_model(X_train_scaled, y_train)

    # Evaluate model
    y_pred = sales_prediction.evaluate_model(X_test_scaled, y_test)

    # Feature importance
    feature_importance = sales_prediction.feature_importance(X)
    print("Feature Importances:\n", feature_importance)

    # Estimate confidence interval
    sales_prediction.estimate_confidence_interval(y_test, y_pred)

    # Save model
    sales_prediction.save_model()

    # Prepare data for LSTM
    X_lstm, y_lstm, scaler = sales_prediction.prepare_lstm_data(data)

    # Build and train LSTM model
    input_shape = (X_lstm.shape[1], X_lstm.shape[2])
    lstm_model = sales_prediction.build_lstm_model(input_shape)
    sales_prediction.train_lstm_model(X_lstm, y_lstm)

    # Save LSTM model
    sales_prediction.save_lstm_model()

if __name__ == '__main__':
    main()
