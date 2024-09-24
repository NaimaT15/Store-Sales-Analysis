import pandas as pd
import numpy as np
import warnings
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller  # For stationarity check
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Autocorrelation
from sklearn.preprocessing import MinMaxScaler  # Scaling for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to merge train/test with store data
def merge_data(train_file, test_file, store_file):
    """
    Merge the train/test and store data based on the 'Store' column.
    If test_file is None, only merge train and store data.
    """
    train_df = pd.read_csv(train_file, dtype={'StateHoliday': 'str'})
    store_df = pd.read_csv(store_file)
    
    train_merged = pd.merge(train_df, store_df, how='inner', on='Store')
    
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        test_merged = pd.merge(test_df, store_df, how='inner', on='Store')
    else:
        test_merged = None  # Return None if test_file is not provided
    
    return train_merged, test_merged


# Preprocessing function
def preprocess_data(train_df, test_df):
    # Handle Missing Values in Numeric Columns
    train_df['CompetitionDistance'].fillna(train_df['CompetitionDistance'].median(), inplace=True)
    test_df['CompetitionDistance'].fillna(0, inplace=True)
    
    train_df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    train_df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    
    train_df['Promo2SinceWeek'].fillna(0, inplace=True)
    train_df['Promo2SinceYear'].fillna(0, inplace=True)
    
    train_df['PromoInterval'].fillna('None', inplace=True)
    test_df['PromoInterval'].fillna('None', inplace=True)

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

    # Drop the original 'Date' column after feature extraction
    train_df.drop(columns=['Date'], inplace=True)
    test_df.drop(columns=['Date'], inplace=True)

    # Convert 'StateHoliday' into numeric values (0, 1, 2, 3)
    def convert_state_holiday(df):
        df['StateHoliday'] = df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
        return df
    
    train_df = convert_state_holiday(train_df)
    test_df = convert_state_holiday(test_df)

    # Handle Missing Values in Categorical Columns (e.g., 'StoreType', 'Assortment')
    train_df['StoreType'].fillna('unknown', inplace=True)
    test_df['StoreType'].fillna('unknown', inplace=True)

    train_df['Assortment'].fillna('unknown', inplace=True)
    test_df['Assortment'].fillna('unknown', inplace=True)

    # Convert categorical columns to numeric using One-Hot Encoding
    def encode_categorical_columns(train_df, test_df):
        categorical_cols = ['StoreType', 'Assortment', 'PromoInterval']
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )

        # For the training dataset, drop 'Sales' and 'Customers'
        train_features = train_df.drop(columns=['Sales', 'Customers'], errors='ignore')
        
        # For the test dataset, drop 'Customers' if it exists
        test_features = test_df.drop(columns=['Customers'], errors='ignore')

        # Fit the encoder on the training data and transform both train and test
        train_df_encoded = preprocessor.fit_transform(train_features)
        test_df_encoded = preprocessor.transform(test_features)

        # Get feature names after encoding
        encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        numeric_cols = train_features.drop(columns=categorical_cols).columns
        final_feature_names = np.concatenate([encoded_cat_names, numeric_cols])

        return train_df_encoded, test_df_encoded, final_feature_names

    train_df_encoded, test_df_encoded, final_feature_names = encode_categorical_columns(train_df, test_df)

    # Convert encoded data back into DataFrame format with proper column names
    train_df_encoded = pd.DataFrame(train_df_encoded, columns=final_feature_names)
    test_df_encoded = pd.DataFrame(test_df_encoded, columns=final_feature_names)

    # Add back the 'Sales' column to the processed train dataset
    if 'Sales' in train_df.columns:
        train_df_encoded['Sales'] = train_df['Sales'].values

    return train_df_encoded, test_df_encoded

# Model building function
def build_model(train_df, target_column):
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    if target_column not in train_df.columns:
        raise KeyError(f"'{target_column}' not found in DataFrame.")
    
    # Define features (X) and target (y)
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Convert all column names in X to strings
    X.columns = X.columns.astype(str)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f'Cross-validation RMSE: {(-cv_scores.mean()) ** 0.5}')

    # Predict on validation data
    y_pred = model_pipeline.predict(X_val)
    val_rmse = rmse(y_val, y_pred)
    print(f'Validation RMSE: {val_rmse}')

    return model_pipeline, X_train, X_val, y_train, y_val

# Prediction function
def prediction():
    train_file = 'C:/Users/Naim/rossmann-store-sales/data/train.csv'
    test_file = 'C:/Users/Naim/rossmann-store-sales/data/test.csv'
    store_file = 'C:/Users/Naim/rossmann-store-sales/data/store.csv'

    train_merged, test_merged = merge_data(train_file, test_file, store_file)


    if 'Sales' not in train_merged.columns:
        raise KeyError("'Sales' column is missing in the merged training data.")
    
    processed_train, processed_test = preprocess_data(train_merged, test_merged)

    if 'Sales' not in processed_train.columns:
        raise KeyError("'Sales' column is missing after preprocessing.")

    # Build and train the model
    model_pipeline, X_train, X_val, y_train, y_val = build_model(processed_train, target_column='Sales')

    # Save the model with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"random_forest_model_{timestamp}.pkl"
    joblib.dump(model_pipeline, model_filename)

    return model_pipeline, X_train, X_val, y_train, y_val


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the RandomForestRegressor model.
    """
    # Extract feature importance values
    feature_importance = model.named_steps['rf'].feature_importances_
    
    # Sort the feature importance values and their corresponding feature names
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot the feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance from Random Forest")
    plt.bar(range(len(feature_names)), feature_importance[indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()
def estimate_confidence_intervals(model, X_val, alpha=0.95):
    """
    Estimate the confidence intervals for predictions from a RandomForestRegressor.
    
    """
    # Get predictions from each tree in the random forest
    all_tree_predictions = np.array([tree.predict(X_val) for tree in model.named_steps['rf'].estimators_])
    
    # Calculate the mean and standard deviation of the predictions
    mean_prediction = np.mean(all_tree_predictions, axis=0)
    std_prediction = np.std(all_tree_predictions, axis=0)
    
    # Calculate the z-score for the desired confidence level
    z = 1.96  # For a 95% confidence interval
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean_prediction - z * std_prediction
    upper_bound = mean_prediction + z * std_prediction
    
    return lower_bound, upper_bound

# Check if the time series is stationary
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary")
    else:
        print("The series is not stationary")
        
# Plot autocorrelation and partial autocorrelation
def plot_acf_pacf(series):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series, lags=50, ax=plt.gca())
    plt.subplot(122)
    plot_pacf(series, lags=50, ax=plt.gca())
    plt.tight_layout()
    plt.show()

# Sliding window transformation for time series
def create_supervised_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Build the LSTM model for time series prediction
def build_lstm_model(X_train, X_val, y_train, y_val, epochs=2, batch_size=32):
    # Reshape the data for LSTM (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_lstm, y_val), verbose=2)
    
    # Make predictions on validation data
    predictions = model.predict(X_val_lstm)
    
    # Evaluate the model using RMSE
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f'Validation RMSE (LSTM): {rmse}')

    return model, predictions

# Prediction function for LSTM with downsampling
def lstm_prediction():
    train_file = 'C:/Users/Naim/rossmann-store-sales/data/train.csv'
    store_file = 'C:/Users/Naim/rossmann-store-sales/data/store.csv'

    # Load and merge data
    train_merged, _ = merge_data(train_file, None, store_file)

    # Convert 'Date' column to datetime and downsample sales data to weekly to reduce memory usage
    train_merged['Date'] = pd.to_datetime(train_merged['Date'])
    sales_series = train_merged.resample('W', on='Date')['Sales'].sum()  # Weekly aggregation

    # Check if the 'Sales' column exists
    if 'Sales' not in train_merged.columns:
        raise KeyError("'Sales' column is missing in the merged training data.")
    
    # Check stationarity
    check_stationarity(sales_series)

    # Plot autocorrelation and partial autocorrelation
    plot_acf_pacf(sales_series)

    # Create supervised learning data using sliding window
    window_size = 30  # Window size of 30 weeks
    X, y = create_supervised_data(sales_series.values, window_size)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data to (-1, 1) range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Build and train the LSTM model
    model, predictions = build_lstm_model(X_train, X_val, y_train, y_val)

    return model, predictions
