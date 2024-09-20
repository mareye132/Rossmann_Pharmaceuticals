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
