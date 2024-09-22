import pandas as pd
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
