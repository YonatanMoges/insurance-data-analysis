


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