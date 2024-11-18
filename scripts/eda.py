import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class EDA:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the DataFrame."""
        self.df = df

    # ============ Univariate Analysis ============
    def plot_numerical_histogram(self, columns: list):
        """Plot histograms for numerical columns."""
        try:
            self.df[columns].hist(bins=15, figsize=(15, 10), layout=(len(columns) // 4 + 1, 4))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_numerical_histogram: {e}")

    def plot_categorical_bar(self, columns: list):
        """Plot bar charts for categorical columns."""
        try:
            for column in columns:
                self.df[column].value_counts().plot(kind='bar', figsize=(10, 5))
                plt.title(f"Bar plot of {column}")
                plt.ylabel('Frequency')
                plt.xlabel(column)
                plt.show()
        except Exception as e:
            print(f"Error in plot_categorical_bar: {e}")

    # ============ Bivariate Analysis ============
    def plot_scatter(self, x_col: str, y_col: str, hue_col: str = None):
        """Scatter plot for two numerical variables with optional hue."""
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue_col, palette='coolwarm')
            plt.title(f"Scatter plot of {x_col} vs {y_col}")
            plt.show()
        except Exception as e:
            print(f"Error in plot_scatter: {e}")

    def plot_correlation_matrix(self, columns: list):
        """Generate a correlation matrix heatmap for numerical columns."""
        try:
            corr = self.df[columns].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.show()
        except Exception as e:
            print(f"Error in plot_correlation_matrix: {e}")

    # ============ Multivariate Analysis ============
    def plot_violin(self, x_col: str, y_col: str, hue: str = None):
        """Violin plot to show distribution and data spread."""
        try:
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=self.df, x=x_col, y=y_col, hue=hue)
            plt.xticks(rotation=45)
            plt.title(f"Violin plot of {y_col} by {x_col}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_violin: {e}")

    def plot_stacked_bar(self, x_col: str, y_col: str, hue_col: str):
        """Plot a stacked bar chart showing the distribution of a categorical variable across another."""
        try:
            grouped_df = self.df.groupby([x_col, hue_col])[y_col].count().unstack().fillna(0)
            grouped_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
            plt.title(f"Stacked Bar Plot of {hue_col} across {x_col}")
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_stacked_bar: {e}")

    # ============ Outlier Detection ============
    def plot_box(self, columns: list):
        """Plot boxplots for outlier detection in numerical data."""
        try:
            self.df[columns].plot(kind='box', subplots=True, layout=(len(columns) // 4 + 1, 4), figsize=(15, 10))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_box: {e}")

    # ============ Advanced Visualization ============
    def plot_correlation_heatmap(self, subset_columns: list = None):
        """Plot a correlation heatmap for a subset of numerical columns."""
        try:
            if subset_columns is None:
                subset_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            corr_matrix = self.df[subset_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plot_correlation_heatmap: {e}")
