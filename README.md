# Insurance Dataset Exploratory Data Analysis (EDA)

## Project Overview
This project performs Exploratory Data Analysis (EDA) on an insurance dataset containing information about insurance policies, client details, car specifications, and payment/claim history. The aim is to uncover trends, relationships, and patterns that can help improve decision-making within the insurance industry.

## Project Structure

├── notebooks/.ipynb files 
├── scripts/.py files  
├── README.md  
└── data

## Dataset Description
The dataset contains the following types of information:

Insurance Policy Details: UnderwrittenCoverID, PolicyID  
Transaction Information: TransactionMonth  
Client Information: IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender  
Location Information: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone  
Car Specifications: ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet  
Plan Information: SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType  
Payment and Claims Information: TotalPremium, TotalClaims

## Features
## EDA
Data Summarization: Descriptive statistics on numerical features such as TotalPremium, TotalClaims, etc.  
Data Quality Assessment: Checks for missing values and proper formatting of categorical variables.  
Univariate, Bivariate, and Multivariate Analysis: Visualizations of distributions and relationships between variables.  
Outlier Detection: Box plots to detect outliers in numerical data.  
Advanced Visualizations: Creative plots to highlight trends and relationships in the data. 

# Insurance Risk Analysis & Predictive Modeling

## Project Overview
This project focuses on analyzing and evaluating the impact of different features on insurance risk and profitability using A/B Hypothesis Testing and Statistical Modeling. The tasks involve performing hypothesis testing to evaluate the effect of different features and building statistical models to predict customer retention, total claims, and premium amounts.


## A/B Hypothesis Testing

### Hypotheses Tested
1. **Risk Differences Across Provinces**  
   - Null Hypothesis (H0): There are no risk differences across provinces.
   
2. **Risk Differences Between Zip Codes**  
   - Null Hypothesis (H0): There are no risk differences between zip codes.
   
3. **Profit Margin Differences Between Zip Codes**  
   - Null Hypothesis (H0): There are no significant margin (profit) differences between zip codes.
   
4. **Risk Differences Between Women and Men**  
   - Null Hypothesis (H0): There are no significant risk differences between women and men.

### Steps:
1. **Select Metrics**  
   - Chose KPIs such as `TotalPremium`, `TotalClaims`, and `ProfitMargin` to evaluate the impact of the features being tested.
   
2. **Data Segmentation**  
   - **Group A (Control Group)**: Plans without the feature.
   - **Group B (Test Group)**: Plans with the feature.
   - Ensured that the two groups (A and B) are statistically equivalent except for the feature being tested.
   
3. **Statistical Testing**  
   - Conducted appropriate statistical tests such as chi-squared tests for categorical variables (e.g., gender, province) and t-tests/z-tests for numerical variables (e.g., profit margins, claims).
   - The significance level was set at 0.05.


## Statistical Modeling

### Steps:

1. **Data Preparation**  
   - **Handling Missing Data**: Imputed missing values where necessary.
   - **Feature Engineering**: Created new features related to `TotalPremium`, `TotalClaims`, and customer behavior.
   - **Encoding Categorical Data**: Applied one-hot encoding for categorical variables.
   - **Train-Test Split**: Split the dataset into 80% training data and 20% test data.

2. **Modeling Techniques**  
   Implemented the following machine learning algorithms:
   - **Linear Regression**
   - **Decision Trees**
   - **Random Forests**
   - **Gradient Boosting (XGBoost)**

3. **Model Evaluation**  
   Evaluated the performance of each model using metrics such as:
   - Accuracy
   - Precision
   - F1-Score
   - RMSE (Root Mean Squared Error) for regression models

4. **Feature Importance Analysis**  
   Used SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to analyze which features were most influential in predicting retention and claims.

# Installation
## Clone this repository:
git clone https://github.com/YonatanMoges/insurance_data_analysis.git

## Install the required dependencies:
pip install -r requirements.txt

# How to Run
Open and run the .ipynb notebook files located in the notebooks/ directory. This will generate visualizations and perform analyses based on the provided dataset.  
Open the .py located in the scripts/ directory to take a look at the modular functions.

## Prerequisites
Python 3.9 and above 
## Libraries 
pandas, numpy, seaborn, matplotlib, shap, lime, xgboost
