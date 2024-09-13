# Insurance Dataset Exploratory Data Analysis (EDA)

Project Overview
This project performs Exploratory Data Analysis (EDA) on an insurance dataset, containing various details about insurance policies, client information, car specifications, and payment/claim history. The analysis focuses on understanding the relationships between these variables and uncovering trends and patterns.

# Project Structure
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   ├── README.md
|   ├── eda.ipynb
│   
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    ├── README.md
    ├── eda.py

# Dataset Description
The dataset contains the following types of information:

Insurance Policy Details: UnderwrittenCoverID, PolicyID
Transaction Information: TransactionMonth
Client Information: IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender
Location Information: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
Car Specifications: ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet
Plan Information: SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType
Payment and Claims Information: TotalPremium, TotalClaims

# Features
Data Summarization: Descriptive statistics on numerical features such as TotalPremium and TotalClaims.
Data Quality Assessment: Checks for missing values and proper formatting of categorical variables.
Univariate, Bivariate, and Multivariate Analysis: Visualizations of distributions and relationships between variables.
Outlier Detection: Use of box plots to detect outliers in numerical data.
Advanced Visualizations: Creative visualizations to highlight key trends and relationships in the data.

# How to Run
Clone this repository.
https://github.com/YonatanMoges/insurance_data_analysis.git
Add the dataset to the data/ folder.
Run the eda.ipynb notebook in notebooks folder.
eda.py for functions in scripts folder.


# Prerequisites
Python 3.9
Jupyter Notebook
Required Libraries: Install using pip
pip install -r requirements.txt
The required packages include:
pandas
numpy
seaborn
matplotlib
