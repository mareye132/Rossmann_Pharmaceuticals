import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configure TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
        df['LastHoliday'] = pd.Timestamp('2024-01-01')
        df['DaysAfterLastHoliday'] = (df['Date'] - df['LastHoliday']).dt.days.clip(lower=0)

        for column in df.select_dtypes(include=['object']).columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        df.fillna(df.median(numeric_only=True), inplace=True)

        X = df.drop(['Sales', 'Date'], axis=1)
        y = df['Sales']
        return X, y

    def check_stationarity(self, df, column='Sales'):
        result = adfuller(df[column])
        return result

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

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    def save_model(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        joblib.dump(self.model_pipeline, f"random_forest_model_{timestamp}.pkl")

    # LSTM methods
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
            tf.keras.layers.LSTM(30, activation='relu', return_sequences=True, input_shape=input_shape),  # Reduced units
            tf.keras.layers.LSTM(30, activation='relu'),  # Reduced units
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        return model

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=16):  # Smaller batch size
        return self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def save_lstm_model(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.lstm_model.save(f"lstm_model_{timestamp}.h5")
