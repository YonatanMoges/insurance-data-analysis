
import matplotlib.pyplot as plt

# Data Summarization

def describe_numerical_columns(df):
    """Generate descriptive statistics for numerical columns"""
    return df.describe()

def check_data_types(df):
    """Check the data types of all columns"""
    return df.dtypes

# Data Quality Assessment

def missing_values(df):
    """Check for missing values in the dataset"""
    return df.isnull().sum()

# Univariate Analysis

def plot_numerical_histogram(df, columns):
    """Plot histograms for numerical columns"""
    df[columns].hist(bins=15, figsize=(15, 10), layout=(len(columns)//4, 4))
    plt.tight_layout()
    plt.show()

def plot_categorical_bar(df, columns):
    """Plot bar charts for categorical columns"""
    for column in columns:
        df[column].value_counts().plot(kind='bar', figsize=(10, 5))
        plt.title(f"Bar plot of {column}")
        plt.ylabel('Frequency')
        plt.xlabel(column)
        plt.show()