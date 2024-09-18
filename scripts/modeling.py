import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data Preprocessing Functions

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

def encode_categorical(df, columns):
    '''One-hot encode or label encode categorical variables'''
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded

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
