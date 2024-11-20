import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Load preprocessed data and model
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Predict function
def predict_premium(input_data, model):
    input_df = pd.DataFrame([input_data])
    return model.predict(input_df)[0]

# Load data and model
data = load_data("../data/processed_data.csv")
model = load_model("../models/random_forest_model.pkl")

# Sidebar - User Navigation
st.sidebar.title("Insurance Dashboard")
section = st.sidebar.radio("Select a section:", ["Data Overview", "Visualizations", "Model Performance", "Predict Premium"])

# --- Section 1: Data Overview ---
if section == "Data Overview":
    st.title("Data Overview")
    st.write("Here's a preview of the insurance dataset.")
    st.dataframe(data.head())
    st.write("### Dataset Statistics")
    st.write(data.describe())

# --- Section 2: Visualizations ---
elif section == "Visualizations":
    st.title("Exploratory Data Analysis")
    st.write("Explore relationships between features.")

    # Select features for visualization
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis:", data.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis:", data.columns)

    # Plot
    fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig)

    # Additional Visualizations
    st.write("### Claims to Premium Ratio Distribution")
    fig2 = px.histogram(data, x="ClaimsToPremiumRatio", nbins=30, title="Claims to Premium Ratio Distribution")
    st.plotly_chart(fig2)

# --- Section 3: Model Performance ---
elif section == "Model Performance":
    st.title("Model Performance")
    st.write("Here are the evaluation metrics for the trained models.")

    # Hard-coded example of model metrics (you can load this dynamically)
    metrics = {
        "RandomForest": {"MSE": 1500.45, "MAE": 25.3, "R2": 0.87},
        "XGBoost": {"MSE": 1350.25, "MAE": 24.1, "R2": 0.89},
    }

    # Display metrics
    for model_name, model_metrics in metrics.items():
        st.write(f"### {model_name}")
        for metric, value in model_metrics.items():
            st.write(f"{metric}: {value:.2f}")

    # Feature Importance
    st.write("### Random Forest Feature Importance")
    feature_importances = model.feature_importances_
    features = data.drop(['TotalPremium', 'TotalClaims'], axis=1).columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_idx = np.argsort(feature_importances)
    plt.barh(features[sorted_idx], feature_importances[sorted_idx], color="skyblue")
    plt.title("Feature Importance")
    st.pyplot(fig)

# --- Section 4: Predict Premium ---
elif section == "Predict Premium":
    st.title("Predict Insurance Premium")
    st.write("Input customer details to predict the insurance premium.")

    # Collect input
    input_data = {
        "Age": st.number_input("Age", min_value=18, max_value=100, value=30),
        "VehicleAge": st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=10),
        "ClaimsToPremiumRatio": st.number_input("Claims to Premium Ratio", min_value=0.0, max_value=10.0, value=0.5),
        "Country_Germany": st.selectbox("Country is Germany?", [0, 1]),
        # Add more features as needed
    }

    if st.button("Predict Premium"):
        prediction = predict_premium(input_data, model)
        st.write(f"### Predicted Premium: ${prediction:.2f}")
