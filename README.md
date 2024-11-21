
# Insurance Data Analysis

This project focuses on analyzing an insurance dataset to extract valuable insights and predict insurance charges based on customer attributes. The ultimate aim is to assist insurance companies in optimizing their pricing strategies and understanding risk factors.

## Project Structure

The repository is organized as follows:

```
├── data/
│   └── [Contains datasets used for analysis and modeling]
├── myenv/
│   └── [Environment setup files or dependencies]
├── notebook/
│   ├── eda.ipynb         # Exploratory Data Analysis
│   ├── hypothesis.ipynb  # Hypothesis testing
│   ├── modeling.ipynb    # Model training and evaluation
│   └── __init__.py       # Indicates this is a Python package
├── scripts/
│   ├── eda.py            # Scripts for EDA tasks
│   ├── hypothesis.py     # Scripts for hypothesis testing
│   ├── modeling.py       # Scripts for model training and evaluation
│   ├── preprocessing.py  # Data preprocessing functions
│   └── __init__.py       # Indicates this is a Python package
├── tests/
│   └── [Unit and integration tests (in progress)]
├── .dvcignore
├── .gitignore
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
```

## Features

1. **Data Preprocessing**: Includes handling missing values, encoding categorical data, and feature scaling.
2. **Exploratory Data Analysis (EDA)**: Detailed visualizations to understand data distributions and relationships.
3. **Hypothesis Testing**: Validating statistical assumptions related to the dataset.
4. **Predictive Modeling**: Building regression models to predict insurance charges.
5. **Dashboard**: Dashboard for overview, visualizations, and prediction.

## Requirements

Install project dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/YonatanMoges/insurance-data-analysis.git
   cd insurance-data-analysis
   ```

2. Set up the Python environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks for analysis.

4. To run the streamlit dashboard:
   - Navigate to the dashboard directory

   ```bash
   cd streamlit
   streamlit run dashboard.py
   ```


