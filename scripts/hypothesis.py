import pandas as pd
import numpy as np
from scipy import stats

class HypothesisTesting:
    def __init__(self, df):
        self.df = df

    def segment_data(self, feature, control_value, test_value):
        """
        Segment the data into control and test groups based on a specified feature.
        """
        control_group = self.df[self.df[feature] == control_value]
        test_group = self.df[self.df[feature] == test_value]
        return control_group, test_group

    def t_test(self, control_group, test_group, kpi):
        """
        Perform a t-test on a KPI between control and test groups.
        Returns the t-statistic and p-value.
        """
        control_values = control_group[kpi].dropna()
        test_values = test_group[kpi].dropna()
        t_stat, p_value = stats.ttest_ind(control_values, test_values, equal_var=False)
        return t_stat, p_value

    def chi_square_test(self, group_col, kpi_col):
        """
        Perform a chi-square test to determine if there's a relationship between group_col and kpi_col.
        Returns the chi2 statistic and p-value.
        """
        contingency_table = pd.crosstab(self.df[group_col], self.df[kpi_col])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2_stat, p_value

    def interpret_results(self, p_value, alpha=0.05):
        """
        Interpret the p-value: Reject or fail to reject the null hypothesis.
        """
        if p_value < alpha:
            return "Reject the null hypothesis (significant difference)"
        else:
            return "Fail to reject the null hypothesis (no significant difference)"

    def test_risk_province(self):
        """
        Test risk differences across provinces using a chi-square test.
        """
        chi2_stat, p_value = self.chi_square_test('Province', 'TotalClaims')
        return p_value

    def test_risk_zipcode(self):
        """
        Test risk differences between zip codes using a chi-square test.
        """
        chi2_stat, p_value = self.chi_square_test('PostalCode', 'TotalClaims')
        return p_value

    def test_margin_zipcode(self):
        """
        Test margin differences between zip codes using a t-test.
        """
        self.df['ProfitMargin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        control_group, test_group = self.segment_data('PostalCode', '12345', '67890')
        t_stat, p_value = self.t_test(control_group, test_group, 'ProfitMargin')
        return p_value

    def test_risk_gender(self):
        """
        Test risk differences between genders using a chi-square test.
        """
        chi2_stat, p_value = self.chi_square_test('Gender', 'TotalClaims')
        return p_value
