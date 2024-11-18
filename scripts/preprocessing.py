import pandas as pd

class Preprocessing:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the DataFrame."""
        self.df = df

    def describe_numerical_columns(self) -> pd.DataFrame:
        """Generate descriptive statistics for numerical columns."""
        try:
            return self.df.describe()
        except Exception as e:
            print(f"Error in describe_numerical_columns: {e}")
            return pd.DataFrame()

    def check_data_types(self) -> pd.Series:
        """Check the data types of all columns."""
        return self.df.dtypes

    def missing_values(self) -> pd.Series:
        """Check for missing values in the dataset."""
        return self.df.isnull().sum()

    def fill_missing_values(self, method='mean'):
        """
        Fill missing values using the specified method.

        Args:
            method (str): Method for filling missing values. Default is 'mean'.
                          Options include 'mean', 'median', and 'mode'.
        """
        # Convert all numerical columns to numeric type
        numeric_columns = self.df.select_dtypes(include='number').columns
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Fill missing values based on the specified method
        if method == 'mean':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        elif method == 'median':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        elif method == 'mode':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mode().iloc[0])
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Missing values filled using method:", method)
    
    def save_to_csv(self, file_path: str):
        """Save the processed DataFrame to a CSV file."""
        try:
            self.df.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")


    def get_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df
