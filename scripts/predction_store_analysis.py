import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Function to merge train/test with store data
def merge_data(train_file, test_file, store_file):
    """
    This function loads the train, test, and store datasets and merges them based on the 'Store' column.
    """
    # Load the datasets
    train_df = pd.read_csv(train_file, dtype={'StateHoliday': 'str'})  # Specify dtype if necessary
    test_df = pd.read_csv(test_file)
    store_df = pd.read_csv(store_file)
    
    # Merge the train and store data on 'Store' column
    train_merged = pd.merge(train_df, store_df, how='inner', on='Store')
    
    # Merge the test and store data on 'Store' column
    test_merged = pd.merge(test_df, store_df, how='inner', on='Store')
    
    return train_merged, test_merged


# Preprocessing function
def preprocess_data(train_df, test_df):
    """
    Preprocess data by handling non-numeric columns, missing values, generating new features, and scaling data.
    """
    
    # Handle Missing Values
    train_df['CompetitionDistance'].fillna(train_df['CompetitionDistance'].median(), inplace=True)
    test_df['CompetitionDistance'].fillna(0, inplace=True)  # Set competition distance to 0 for test
    
    train_df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    train_df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    
    train_df['Promo2SinceWeek'].fillna(0, inplace=True)
    train_df['Promo2SinceYear'].fillna(0, inplace=True)
    
    train_df['PromoInterval'].fillna('None', inplace=True)
    
    # Feature Engineering: Extract new features from datetime columns
    def create_date_features(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        return df

    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)

    # Convert categorical columns to numeric
    def encode_categorical_columns(df):
        df['StateHoliday'] = df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
        df['SchoolHoliday'] = df['SchoolHoliday'].astype(int)
        return df

    train_df = encode_categorical_columns(train_df)
    test_df = encode_categorical_columns(test_df)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
                      'Year', 'Month', 'Day', 'WeekOfYear']
    
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    return train_df, test_df


# Main Function to call everything
def main():
    train_file = r'C:/Users/Naim/rossmann-store-sales/data/train.csv'  # Use full file paths
    test_file = r'C:/Users/Naim/rossmann-store-sales/data/test.csv'
    store_file = r'C:/Users/Naim/rossmann-store-sales/data/store.csv'

    # Merge the data
    train_merged, test_merged = merge_data(train_file, test_file, store_file)

    # Preprocess the data
    processed_train, processed_test = preprocess_data(train_merged, test_merged)

    # Display the processed data
    print("Processed Train Data Sample:")
    print(processed_train.head())

    print("Processed Test Data Sample:")
    print(processed_test.head())

# Call the main function to execute the script
# if __name__ == "__main__":
    main()
