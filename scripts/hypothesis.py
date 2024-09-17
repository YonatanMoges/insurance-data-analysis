import pandas as pd
import numpy as np
from scipy import stats

# Load and preprocess data
def load_data(file_path):
    """Load the dataset from a CSV file and preprocess it."""
    data = pd.read_csv(file_path)
    # Further preprocessing steps, such as handling missing values, could go here
    return data

# Data Cleaning
def clean_data(data):
    """
    Clean the dataset by handling missing values, correcting data types, and removing duplicates.
    """
    # Drop duplicates if any
    data = data.drop_duplicates()
    
    # Fill missing values for categorical columns with 'Unknown' or the most frequent value
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna('Unknown')

    # Fill missing values for numerical columns with mean or median
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Convert date columns to datetime if needed
    if 'TransactionMonth' in data.columns:
        data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
    
    return data

# Function to segment the data into control and test groups
def segment_data(data, feature, control_value, test_value):
    """
    Segment the data into control and test groups based on the feature.
    Control group: feature == control_value
    Test group: feature == test_value
    """
    control_group = data[data[feature] == control_value]
    test_group = data[data[feature] == test_value]
    return control_group, test_group


from scipy import stats
def t_test(control_group, test_group, kpi):
    """
    Perform a t-test on a KPI between control and test groups.
    Returns the t-statistic and p-value.
    """
    control_values = control_group[kpi].dropna()
    test_values = test_group[kpi].dropna()
    
    t_stat, p_value = stats.ttest_ind(control_values, test_values, equal_var=False)
    return t_stat, p_value


# Perform chi-squared test for categorical data
def chi_square_test(data, group_col, kpi_col):
    """
    Perform a chi-square test to determine if there's a relationship between the group_col and kpi_col.
    """
    contingency_table = pd.crosstab(data[group_col], data[kpi_col])
    chi2_stat, p_value, dof, ex = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value

# Perform hypothesis testing for risk differences across provinces
def test_risk_province(data):
    control_group, test_group = segment_data(data, 'Province', 'Province_A', 'Province_B')
    chi2_stat, p_value = chi_square_test(data, 'Province', 'TotalClaims')
    return p_value


# Perform hypothesis testing for risk differences between zip codes
def test_risk_zipcode(data):
    control_group, test_group = segment_data(data, 'PostalCode', '12345', '67890')
    chi2_stat, p_value = chi_square_test(data, 'PostalCode', 'TotalClaims')
    return p_value


# Perform hypothesis testing for margin differences between zip codes
def test_margin_zipcode(data):
    data['ProfitMargin'] = data['TotalPremium'] - data['TotalClaims']
    control_group, test_group = segment_data(data, 'PostalCode', '12345', '67890')
    t_stat, p_value = t_test(control_group, test_group, 'ProfitMargin')
    return p_value


# Perform hypothesis testing for risk differences between genders
def test_risk_gender(data):
    control_group, test_group = segment_data(data, 'Gender', 'Male', 'Female')
    chi2_stat, p_value = chi_square_test(data, 'Gender', 'TotalClaims')
    return p_value


def interpret_results(p_value, alpha=0.05):
    """
    Interpret the p-value: Reject or fail to reject the null hypothesis.
    """
    if p_value < alpha:
        return "Reject the null hypothesis (significant difference)"
    else:
        return "Fail to reject the null hypothesis (no significant difference)"
