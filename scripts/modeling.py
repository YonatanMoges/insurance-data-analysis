import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

class ModelingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }

    # --- DATA PROCESSING FUNCTIONS ---
    def preprocess_data(self, X):
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            elif X[col].apply(type).nunique() > 1:  # Mixed types
                X[col] = X[col].astype(str).fillna("Unknown")
                X[col] = LabelEncoder().fit_transform(X[col])
        X.fillna(0, inplace=True)  # Handle missing values
        return X

    def load_and_clean_data(self, filepath):
        dtype_spec = {32: 'object', 37: 'object'}
        data = pd.read_csv(filepath, dtype=dtype_spec)
        data = data.drop_duplicates(keep="first")
        return data

    def handle_missing_data(self, df):
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].mean())
        return df

    def feature_engineering(self, df):
        df['VehicleAge'] = 2023 - df['RegistrationYear']
        df['ClaimsToPremiumRatio'] = df['TotalClaims'] / df['TotalPremium']
        df['ClaimsToPremiumRatio'] = df['ClaimsToPremiumRatio'].replace([np.inf, -np.inf], 0).fillna(0)
        return df

    def process_dates(self, df, date_column='TransactionMonth'):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df['TransactionYear'] = df[date_column].dt.year
        df['TransactionMonthOnly'] = df[date_column].dt.month
        df = df.drop(columns=[date_column])
        return df

    def encode_data(self, df, columns_label=None, columns_onehot=None):
        df_copy = df.copy()
        if columns_label:
            for col in columns_label:
                label = LabelEncoder()
                df_copy[col] = label.fit_transform(df_copy[col].astype(str))
        if columns_onehot:
            df_copy = pd.get_dummies(data=df_copy, columns=columns_onehot, drop_first=True)
        return df_copy

    def scale_data(self, X_train, X_test, numeric_columns):
        X_train[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        return X_train, X_test

    # --- TRAINING AND EVALUATION ---
    def train_model(self, model_name, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        model = self.models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"Model {model_name} trained in {time.time() - start_time:.2f} seconds.")
        return model

    def predict_model(self, model, X_test):
        X_test = self.preprocess_data(X_test)
        return model.predict(X_test)

    def evaluate_model(self, y_true, y_pred):
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    # --- VISUALIZATION AND DISPLAY ---
    def plot_feature_importance(self, model, feature_names, top_n=10):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            plt.figure(figsize=(8, 6))
            plt.barh(np.array(feature_names)[indices], importances[indices], color="skyblue")
            plt.title(f"Top {top_n} Feature Importances")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.show()
        else:
            print("The model does not support feature importance plotting.")

    def display_results(self, results):
        print("\nModel Evaluation Metrics:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    # --- MODEL PERSISTENCE ---
    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}.")
        return model
