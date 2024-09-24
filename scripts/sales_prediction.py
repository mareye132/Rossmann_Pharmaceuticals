import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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


    def load_data(self):
        """
        Loads store and train data from CSV files.
        """
        store_df = pd.read_csv(self.store_data_path)
        train_df = pd.read_csv(self.train_data_path)
        df = pd.merge(train_df, store_df, on='Store')
        return df

    def preprocess_data(self, df):
        """
        Preprocess the data for training the model.
        Handles missing values and encodes categorical features.
        """
        # Handling missing values
        df.fillna(0, inplace=True)

        # Encoding categorical variables (if any)
        df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes
        df['StoreType'] = df['StoreType'].astype('category').cat.codes
        df['Assortment'] = df['Assortment'].astype('category').cat.codes

        # Feature selection
        X = df[['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']]

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

        """
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        """
        result = adfuller(df[column])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print(f'Critical Value {key}: {value}')
        return result

    def plot_acf_pacf(self, df, column='Sales', lags=40):
        """
        Plots ACF and PACF for a given time series column.
        
        Parameters:
        df (DataFrame): The DataFrame containing the time series data.
        column (str): The name of the column for which to plot ACF and PACF.
        lags (int): Number of lags to include in the plot.
        """
        plt.figure(figsize=(12, 6))

        # Plot ACF
        plt.subplot(121)
        plot_acf(df[column], lags=lags, ax=plt.gca())
        plt.title('Autocorrelation Function (ACF)')

        # Plot PACF
        plt.subplot(122)
        plot_pacf(df[column], lags=lags, ax=plt.gca())
        plt.title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        plt.show()

    def split_and_scale_data(self, X, y):
        """
        Splits the data into training and testing sets and scales the features.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_and_train_model(self, X_train, y_train):
        """
        Builds and trains the RandomForest model.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model and calculates the Mean Squared Error.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    def feature_importance(self, X):
        """
        Returns a DataFrame containing feature importance scores.
        """
        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df

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

