import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from clairvoyants.disaggregation import prepare_history_to_disaggregate


class TestPrepareHistoryToDisaggregate:
    """Test cases for prepare_history_to_disaggregate function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create sample unaggregated history (14 days)
        dates = pd.date_range('2022-01-01', periods=14, freq='D')
        actual = [10, 12, 8, 15, 11, 9, 13,  # Week 1: sum = 78
                  14, 16, 10, 18, 12, 8, 15]  # Week 2: sum = 93
        self.history = pd.DataFrame({
            'dt': dates,
            'actual': actual
        })
        
        # Create aggregated history (2 weeks)
        agg_dates = pd.date_range('2022-01-07', periods=2, freq='7D')
        self.aggregated_history = pd.DataFrame({
            'dt': agg_dates,
            'actual': [78, 93]  # Weekly totals
        })
        
        # Create external regressors
        self.x_reg = pd.DataFrame({
            'dt': dates,
            'marketing_spend': [100, 120, 80, 150, 110, 90, 130,
                               140, 160, 100, 180, 120, 80, 150],
            'holiday_indicator': [0, 0, 0, 0, 0, 0, 1,  # Weekend
                                 0, 0, 0, 0, 0, 0, 1]   # Weekend
        })
    
    def test_basic_functionality(self):
        """Test basic functionality with no external regressors."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7
        )
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14  # Same length as input
        assert 'proportion_aggregated' in result.columns
        assert 'logit_proportion_aggregated' in result.columns
        assert 'aggregated_actual' in result.columns
        
        # Check proportions sum to 1 for each week
        week1_props = result['proportion_aggregated'][:7]
        week2_props = result['proportion_aggregated'][7:]
        
        assert abs(week1_props.sum() - 1.0) < 1e-10
        assert abs(week2_props.sum() - 1.0) < 1e-10
        
        # Check aggregated_actual is repeated correctly
        assert all(result['aggregated_actual'][:7] == 78)
        assert all(result['aggregated_actual'][7:] == 93)
    
    def test_with_external_regressors(self):
        """Test functionality with external regressors."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7,
            x_reg=self.x_reg[['marketing_spend', 'holiday_indicator']],
            x_cols_seasonal_interactions=['marketing_spend']
        )
        
        # Check that external regressors are included
        assert 'marketing_spend' in result.columns
        assert 'holiday_indicator' in result.columns
        
        # Check interaction terms are created
        interaction_cols = [col for col in result.columns 
                           if col.startswith('marketing_spend_x_p')]
        assert len(interaction_cols) > 0
        
        # Check dummy variables for period
        dummy_cols = [col for col in result.columns if col.startswith('p_')]
        assert len(dummy_cols) == 7  # One for each day of week
    
    def test_proportion_bounds(self):
        """Test that proportions are bounded between 1e-6 and 1-1e-6."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7
        )
        
        proportions = result['proportion_aggregated']
        assert all(proportions >= 1e-6)
        assert all(proportions <= 1 - 1e-6)
    
    def test_logit_calculation(self):
        """Test that logit transformation is correct."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7
        )
        
        # Manual logit calculation
        p = result['proportion_aggregated']
        expected_logit = np.log(p / (1 - p))
        
        np.testing.assert_array_almost_equal(
            result['logit_proportion_aggregated'], 
            expected_logit
        )
    
    def test_period_dummy_variables(self):
        """Test that period dummy variables are correctly created."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7
        )
        
        # Check that p column is created and then dropped
        assert 'p' not in result.columns
        
        # Check dummy variables for each period
        dummy_cols = [col for col in result.columns if col.startswith('p_')]
        assert len(dummy_cols) == 7  # p_1 through p_7
        
        # Check that each row has exactly one dummy variable = 1
        dummy_data = result[dummy_cols]
        assert all(dummy_data.sum(axis=1) == 1)
    
    def test_invalid_history_length(self):
        """Test that function raises error for invalid history length."""
        # Create history with wrong length (13 days instead of 14)
        invalid_history = self.history.iloc[:-1]
        
        with pytest.raises(Exception, match='history must be an integer number of period_agg'):
            prepare_history_to_disaggregate(
                invalid_history,
                self.aggregated_history,
                period_agg=7
            )
    
    def test_edge_case_zero_actual(self):
        """Test handling of zero actual values."""
        # Create history with some zero values
        history_with_zeros = self.history.copy()
        history_with_zeros.loc[0, 'actual'] = 0
        
        result = prepare_history_to_disaggregate(
            history_with_zeros,
            self.aggregated_history,
            period_agg=7
        )
        
        # Should still work due to bounds checking
        assert all(result['proportion_aggregated'] >= 1e-6)
        assert all(result['proportion_aggregated'] <= 1 - 1e-6)
    
    def test_different_period_agg(self):
        """Test with different aggregation period."""
        # Create 6-day history and 2-period aggregated data
        history_6 = self.history.iloc[:6]
        agg_6 = pd.DataFrame({
            'dt': pd.date_range('2022-01-03', periods=2, freq='3D'),
            'actual': [30, 33]  # 3-day totals
        })
        
        result = prepare_history_to_disaggregate(
            history_6,
            agg_6,
            period_agg=3
        )
        
        assert len(result) == 6
        assert len([col for col in result.columns if col.startswith('p_')]) == 3
    
    def test_no_external_regressors(self):
        """Test behavior when x_reg is None."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7,
            x_reg=None
        )
        
        # Should still create period dummy variables
        dummy_cols = [col for col in result.columns if col.startswith('p_')]
        assert len(dummy_cols) == 7
    
    def test_empty_seasonal_interactions(self):
        """Test with empty seasonal interactions list."""
        result = prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7,
            x_reg=self.x_reg[['marketing_spend']],
            x_cols_seasonal_interactions=[]
        )
        
        # Should include x_reg columns but no interactions
        assert 'marketing_spend' in result.columns
        interaction_cols = [col for col in result.columns 
                           if 'x_p' in col]
        assert len(interaction_cols) == 0
    
    def test_data_integrity(self):
        """Test that input data is not modified."""
        original_history = self.history.copy()
        original_agg = self.aggregated_history.copy()
        
        prepare_history_to_disaggregate(
            self.history,
            self.aggregated_history,
            period_agg=7
        )
        
        # Check that original dataframes are unchanged
        pd.testing.assert_frame_equal(original_history, self.history)
        pd.testing.assert_frame_equal(original_agg, self.aggregated_history)
