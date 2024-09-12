
import matplotlib.pyplot as plt
import seaborn as sns

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

# Bivariate Analysis

def plot_scatter(df, x_col, y_col, hue=None):
    """Scatter plot for bivariate analysis"""
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
    plt.title(f"Scatter plot of {x_col} vs {y_col}")
    plt.show()

def correlation_matrix(df, columns):
    """Generate correlation matrix and plot heatmap"""
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# Data Comparison Over Geography

def plot_geographic_trend(df, x_col, y_col, hue=None):
    """Plot trends over geography"""
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue)
    plt.title(f"Trend of {y_col} over {x_col}")
    plt.show()

# Outlier Detection

def box_plot(df, columns):
    """Plot boxplots for outlier detection in numerical data"""
    df[columns].plot(kind='box', subplots=True, layout=(len(columns)//4, 4), figsize=(15, 10))
    plt.tight_layout()
    plt.show()

