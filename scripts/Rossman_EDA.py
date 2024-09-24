import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
logging.basicConfig(filename='eda.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(store_data_path, train_data_path):
    """Load store and training datasets."""
    try:
        store_df = pd.read_csv(store_data_path)
        train_df = pd.read_csv(train_data_path)
        logging.info("Data loaded successfully.")
        return store_df, train_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def merge_data(store_df, train_df):
    """Merge store and training datasets."""
    try:
        merged_df = pd.merge(train_df, store_df, on='Store', how='left')
        logging.info("Data merged successfully.")
        return merged_df
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def clean_data(df):
    """Clean the dataset by handling missing values and filtering rows."""
    try:
        df = df.dropna(subset=['Sales'])  # Example of removing rows with missing sales
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df.fillna(0, inplace=True)
        logging.info("Data cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

def plot_promo_sales_distribution(df):
    """Plot distribution of sales during promotions."""
    try:
        sns.boxplot(x='Promo', y='Sales', data=df)
        plt.title('Sales Distribution During Promotion')
        plt.show()
        logging.info("Plotted sales distribution during promotions.")
    except Exception as e:
        logging.error(f"Error plotting promo sales distribution: {e}")

def plot_holiday_sales(df):
    """Plot sales behavior before, during, and after holidays."""
    try:
        sns.boxplot(x='StateHoliday', y='Sales', data=df)
        plt.title('Sales Distribution During Holidays')
        plt.show()
        logging.info("Plotted holiday sales distribution.")
    except Exception as e:
        logging.error(f"Error plotting holiday sales: {e}")

def plot_assortment_sales(df):
    """Plot the effect of assortment type on sales."""
    try:
        assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()
        sns.barplot(data=assortment_sales, x='Assortment', y='Sales')
        plt.title('Effect of Assortment Type on Sales')
        plt.show()
        logging.info("Plotted assortment effect on sales.")
    except Exception as e:
        logging.error(f"Error plotting assortment sales: {e}")

def plot_competition_distance_sales(df):
    """Plot the effect of competitor distance on sales."""
    try:
        sns.scatterplot(data=df, x='CompetitionDistance', y='Sales')
        plt.title('Effect of Competitor Distance on Sales')
        plt.show()
        logging.info("Plotted competitor distance effect on sales.")
    except Exception as e:
        logging.error(f"Error plotting competition distance sales: {e}")

def save_cleaned_data(df, output_path):
    """Save cleaned data to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving cleaned data: {e}")
        raise

if __name__ == "__main__":
    # File paths
    store_data_path = "C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/Data/store.csv"
    train_data_path = "C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/Data/train.csv"
    output_path = "C:/Users/user/Desktop/Github/Rossmann_Pharmaceuticals/Data/cleaned_data.csv"
    
    # Load and merge data
    store_df, train_df = load_data(store_data_path, train_data_path)
    merged_df = merge_data(store_df, train_df)
    
    # Clean the data
    merged_df = clean_data(merged_df)
    
    # Perform EDA
    plot_promo_sales_distribution(merged_df)
    plot_holiday_sales(merged_df)
    plot_assortment_sales(merged_df)
    plot_competition_distance_sales(merged_df)
    
    # Save the cleaned dataset
    save_cleaned_data(merged_df, output_path)

import numpy as np

def load_data(store_data_path, train_data_path):
    store_df = pd.read_csv(store_data_path)
    train_df = pd.read_csv(train_data_path, low_memory=False)
    return store_df, train_df

def handle_missing_values(store_df):
    store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median())
    store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna(0)
    store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna(0)
    store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(0)
    store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(0)
    store_df['PromoInterval'] = store_df['PromoInterval'].fillna(0)
    return store_df

def merge_data(train_df, store_df):
    merged_df = pd.merge(train_df, store_df, how='left', on='Store')
    return merged_df

def analyze_promotions(merged_df):
    merged_df['StateHoliday'] = merged_df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 1, 'c': 1})
    merged_df['StateHoliday'] = merged_df['StateHoliday'].astype(int)
    merged_df['HolidayPeriod'] = 'No Holiday'
    
    # Mark holiday periods
    merged_df.loc[merged_df['StateHoliday'] == 1, 'HolidayPeriod'] = 'During Holiday'
    merged_df.loc[merged_df['StateHoliday'].shift(-1) == 1, 'HolidayPeriod'] = 'Before Holiday'
    merged_df.loc[merged_df['StateHoliday'].shift(1) == 1, 'HolidayPeriod'] = 'After Holiday'
    
    # Group by holiday period to find average sales
    holiday_sales = merged_df.groupby('HolidayPeriod')['Sales'].mean().reset_index()
    return holiday_sales

def seasonal_analysis(merged_df):
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
    
    christmas_sales = merged_df[merged_df['Month'] == 12]
    easter_sales = merged_df[(merged_df['Month'] == 4) & (merged_df['DayOfWeek'] == 6)]
    
    return christmas_sales, easter_sales

