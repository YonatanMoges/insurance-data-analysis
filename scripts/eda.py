
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

# Bivariate Analysis as a function of zipcode

def plot_scatter_by_zipcode(df, x_col, y_col, zipcode_col):
    """Scatter plot of TotalPremium vs TotalClaims as a function of ZipCode"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=zipcode_col, palette='coolwarm')
    plt.title(f"Scatter plot of {x_col} vs {y_col} by {zipcode_col}")
    plt.legend(title=zipcode_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def correlation_matrix(df, columns):
    """Generate correlation matrix and plot heatmap"""
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# Data Comparison Over Geography

def plot_trend_over_geography(df, x_col, y_col, hue, geography_col):
    """Plot trends over geography, comparing different features"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, ci=None)
    plt.title(f"Trend of {y_col} by {hue} over {x_col}")
    plt.xlabel(geography_col)
    plt.ylabel(y_col)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Outlier Detection

def box_plot(df, columns):
    """Plot boxplots for outlier detection in numerical data"""
    df[columns].plot(kind='box', subplots=True, layout=(len(columns)//4, 4), figsize=(15, 10))
    plt.tight_layout()
    plt.show()

# Advanced Visualization

# Heatmap of correlations for a subset of columns
def plot_correlation_heatmap(df, subset_columns=None):
    """Plot a correlation heatmap for a subset of numerical columns."""
    if subset_columns is None:
        subset_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
    
    corr_matrix = df[subset_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Violin plot for TotalPremium by VehicleType
def plot_violin(df, x_col, y_col, hue=None):
    """Violin plot to show distribution and data spread"""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(f"Violin plot of {y_col} by {x_col}")
    plt.tight_layout()
    plt.show()

# Stacked Bar Plot of categorical columns
def plot_stacked_bar(df, x_col, y_col, hue_col):
    """Plot a stacked bar chart showing the distribution of a categorical variable across another."""
    grouped_df = df.groupby([x_col, hue_col])[y_col].count().unstack().fillna(0)
    
    # Plot the stacked bar chart
    grouped_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title(f"Stacked Bar Plot of {hue_col} across {x_col}")
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


