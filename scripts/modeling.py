import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data Preprocessing Functions
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    # Removing duplicates
    data = data.drop_duplicates(keep="first")
    return data

def handle_missing_data(df):
    '''Handle missing data by imputing or dropping based on the column type'''
    # Example: Fill numeric columns with the mean, categorical with the mode
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

def feature_engineering(df):
    '''Create new features like vehicle age, claims ratio, etc.'''
    # Example feature: Vehicle age
    df['VehicleAge'] = 2023 - df['RegistrationYear']
    
    # Example feature: Claims to premium ratio
    df['ClaimsToPremiumRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['ClaimsToPremiumRatio'] = df['ClaimsToPremiumRatio'].replace([np.inf, -np.inf], 0)
    df['ClaimsToPremiumRatio'] = df['ClaimsToPremiumRatio'].fillna(0)
    
    return df



import pandas as pd

import pandas as pd

def encode_categorical(df, columns):
    '''One-hot encode categorical variables and ensure all encoded columns are integers'''
    # Check if the columns exist in the DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing in the DataFrame: {missing_cols}")
    
    # Use get_dummies for one-hot encoding
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    
    df_encoded = df_encoded.dropna()

    
    # Convert any boolean columns to integers (0 and 1)
    df_encoded = df_encoded.astype(int)
    
    return df_encoded






def encoder(method, dataframe, columns_label, columns_onehot):
    if method == 'labelEncoder':      
        df_lbl = dataframe.copy()
        for col in columns_label:
            label = LabelEncoder()
            label.fit(list(dataframe[col].values))
            df_lbl[col] = label.transform(df_lbl[col].values)
        return df_lbl
    
    elif method == 'oneHotEncoder':
        df_oh = dataframe.copy()
        df_oh= pd.get_dummies(data=df_oh, prefix='ohe', prefix_sep='_',
                       columns=columns_onehot, drop_first=True, dtype='int8')
        return df_oh

def process_dates(df):
    '''Convert date columns to datetime and extract useful features'''
    # Example: Convert the 'TransactionMonth' to datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Extract year, month, day, or other relevant features
    df['TransactionYear'] = df['TransactionMonth'].dt.year
    df['TransactionMonthOnly'] = df['TransactionMonth'].dt.month

    # Drop the original 'TransactionMonth' column if no longer needed
    df = df.drop(columns=['TransactionMonth'])

    return df

# Modeling Functions

def train_linear_regression(X_train, y_train):
    '''Train a linear regression model'''
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    '''Train a random forest model'''
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    '''Train an XGBoost model'''
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluation Functions

def evaluate_model(y_true, y_pred):
    '''Evaluate model performance using regression metrics'''
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R2': r2}


