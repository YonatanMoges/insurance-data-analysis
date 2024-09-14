# Insurance Dataset Exploratory Data Analysis (EDA)

## Project Overview
This project performs Exploratory Data Analysis (EDA) on an insurance dataset containing information about insurance policies, client details, car specifications, and payment/claim history. The aim is to uncover trends, relationships, and patterns that can help improve decision-making within the insurance industry.

## Project Structure

├── notebooks/eda.ipynb  
├── scripts/eda.py  
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
Data Summarization: Descriptive statistics on numerical features such as TotalPremium, TotalClaims, etc.  
Data Quality Assessment: Checks for missing values and proper formatting of categorical variables.  
Univariate, Bivariate, and Multivariate Analysis: Visualizations of distributions and relationships between variables.  
Outlier Detection: Box plots to detect outliers in numerical data.  
Advanced Visualizations: Creative plots to highlight trends and relationships in the data.  

# Installation
## Clone this repository:
git clone https://github.com/YonatanMoges/insurance_data_analysis.git

## Install the required dependencies:
pip install -r requirements.txt

# How to Run
Open and run the eda.ipynb notebook located in the notebooks/ directory. This will generate visualizations and perform analyses based on the provided dataset.  
Open eda.py located in the scripts/ directory to take a look at the modular functions.

## Prerequisites
Python 3.9 and above 
## Libraries 
pandas, numpy, seaborn, matplotlib
