import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class ModelingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }

    # Data Preprocessing
    def preprocess_data(self, X):
        """
        Preprocesses the input data by encoding categorical features.

        Parameters:
        X (pd.DataFrame): The data to preprocess.

        Returns:
        pd.DataFrame: Preprocessed data.
        """
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            elif X[col].apply(type).nunique() > 1:  # Mixed types
                X[col] = X[col].astype(str).fillna("Unknown")
                X[col] = LabelEncoder().fit_transform(X[col])

        X.fillna(0, inplace=True)  # Handle missing values
        return X

    def load_and_clean_data(self, filepath):
        dtype_spec={32: 'object', 37:'object'}
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

    # Training and Evaluation
    def train_model(self, model_name, X_train, y_train):
        """
        Trains the specified model on the provided training data.

        Parameters:
        model_name (str): Name of the model to train.
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series or np.array): Target variable.

        Returns:
        sklearn model: Trained model instance.
        """
        # Preprocess X_train
        X_train = self.preprocess_data(X_train)

        # Train model
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def predict_model(self, model, X_test):
        """
        Predicts using the trained model and preprocessed test data.

        Parameters:
        model (sklearn model): Trained model.
        X_test (pd.DataFrame): Features for prediction.

        Returns:
        np.array: Predictions.
        """
        # Preprocess X_test
        X_test = self.preprocess_data(X_test)
        return model.predict(X_test)
        
        
    def evaluate_model(self, y_true, y_pred):
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    # Visualization and Display
    def plot_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(len(feature_names)), importances[indices], align="center")
            plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
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
