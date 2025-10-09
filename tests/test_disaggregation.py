import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from clairvoyants.disaggregation import (
    prepare_history_to_disaggregate,
    prepare_forecast_to_disaggregate,
    disaggregate_forecast
)


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


class TestPrepareForecastToDisaggregate:
    """Test cases for prepare_forecast_to_disaggregate function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create forecast dates (14 days)
        self.forecast_dates = pd.date_range('2022-01-15', periods=14, freq='D')
        
        # Create aggregated forecast (2 weeks)
        agg_dates = pd.date_range('2022-01-21', periods=2, freq='7D')
        self.aggregated_forecast = pd.DataFrame({
            'dt': agg_dates,
            'forecast': [85, 95],
            'forecast_lower': [75, 85],
            'forecast_upper': [95, 105]
        })
        
        # Create future external regressors
        self.x_future = pd.DataFrame({
            'dt': self.forecast_dates,
            'marketing_spend': [110, 130, 90, 160, 120, 100, 140,
                               150, 170, 110, 190, 130, 90, 160],
            'holiday_indicator': [0, 0, 0, 0, 0, 0, 1,  # Weekend
                                 0, 0, 0, 0, 0, 0, 1]   # Weekend
        })
    
    def test_basic_functionality(self):
        """Test basic functionality with no external regressors."""
        result = prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
            period_agg=7
        )
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14  # Same length as forecast dates
        assert 'aggregated_forecast' in result.columns
        assert 'aggregated_forecast_lower' in result.columns
        assert 'aggregated_forecast_upper' in result.columns
        assert 'dt' in result.columns
        
        # Check aggregated values are repeated correctly
        assert all(result['aggregated_forecast'][:7] == 85)
        assert all(result['aggregated_forecast'][7:] == 95)
        assert all(result['aggregated_forecast_lower'][:7] == 75)
        assert all(result['aggregated_forecast_upper'][:7] == 95)
    
    def test_with_external_regressors(self):
        """Test functionality with external regressors."""
        result = prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
            period_agg=7,
            x_future=self.x_future[['marketing_spend', 'holiday_indicator']],
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
    
    def test_period_dummy_variables(self):
        """Test that period dummy variables are correctly created."""
        result = prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
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
    
    def test_invalid_forecast_dates_length(self):
        """Test that function raises error for invalid forecast dates length."""
        # Create forecast dates with wrong length (13 days instead of 14)
        invalid_dates = self.forecast_dates[:-1]
        
        with pytest.raises(Exception, match='forecast dates must be an integer number of period_agg'):
            prepare_forecast_to_disaggregate(
                invalid_dates,
                self.aggregated_forecast,
                period_agg=7
            )
    
    def test_different_period_agg(self):
        """Test with different aggregation period."""
        # Create 6-day forecast dates and 2-period aggregated forecast
        forecast_6 = self.forecast_dates[:6]
        agg_6 = pd.DataFrame({
            'dt': pd.date_range('2022-01-17', periods=2, freq='3D'),
            'forecast': [45, 50],
            'forecast_lower': [40, 45],
            'forecast_upper': [50, 55]
        })
        
        result = prepare_forecast_to_disaggregate(
            forecast_6,
            agg_6,
            period_agg=3
        )
        
        assert len(result) == 6
        assert len([col for col in result.columns if col.startswith('p_')]) == 3
    
    def test_no_external_regressors(self):
        """Test behavior when x_future is None."""
        result = prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
            period_agg=7,
            x_future=None
        )
        
        # Should still create period dummy variables
        dummy_cols = [col for col in result.columns if col.startswith('p_')]
        assert len(dummy_cols) == 7
    
    def test_empty_seasonal_interactions(self):
        """Test with empty seasonal interactions list."""
        result = prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
            period_agg=7,
            x_future=self.x_future[['marketing_spend']],
            x_cols_seasonal_interactions=[]
        )
        
        # Should include x_future columns but no interactions
        assert 'marketing_spend' in result.columns
        interaction_cols = [col for col in result.columns 
                           if 'x_p' in col]
        assert len(interaction_cols) == 0
    
    def test_data_integrity(self):
        """Test that input data is not modified."""
        original_forecast = self.aggregated_forecast.copy()
        original_x_future = self.x_future.copy()
        
        prepare_forecast_to_disaggregate(
            self.forecast_dates,
            self.aggregated_forecast,
            period_agg=7,
            x_future=self.x_future
        )
        
        # Check that original dataframes are unchanged
        pd.testing.assert_frame_equal(original_forecast, self.aggregated_forecast)
        pd.testing.assert_frame_equal(original_x_future, self.x_future)


class TestDisaggregateForecast:
    """Test cases for disaggregate_forecast function."""
    
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
        
        # Create aggregated forecast
        fcst_dates = pd.date_range('2022-01-21', periods=2, freq='7D')
        self.aggregated_forecast = pd.DataFrame({
            'dt': fcst_dates,
            'forecast': [85, 95],
            'forecast_lower': [75, 85],
            'forecast_upper': [95, 105]
        })
        
        # Create external regressors for history
        self.x_reg = pd.DataFrame({
            'dt': dates,
            'marketing_spend': [100, 120, 80, 150, 110, 90, 130,
                               140, 160, 100, 180, 120, 80, 150]
        })
        
        # Create external regressors for forecast
        fcst_dates_daily = pd.date_range('2022-01-15', periods=14, freq='D')
        self.x_future = pd.DataFrame({
            'dt': fcst_dates_daily,
            'marketing_spend': [110, 130, 90, 160, 120, 100, 140,
                               150, 170, 110, 190, 130, 90, 160]
        })
    
    def test_basic_disaggregation(self):
        """Test basic disaggregation without external regressors."""
        result = disaggregate_forecast(
            self.history,
            self.aggregated_history,
            self.aggregated_forecast,
            dt_units='D',
            period_agg=7,
            period_disagg=1
        )
        
        # Check output structure
        assert isinstance(result, dict)
        assert 'disaggregated' in result
        assert 'coefficients' in result
        assert 'residuals' in result
        assert 'summary' in result
        
        # Check disaggregated forecast
        disagg = result['disaggregated']
        assert isinstance(disagg, pd.DataFrame)
        assert 'dt' in disagg.columns
        assert 'forecast' in disagg.columns
        assert 'forecast_lower' in disagg.columns
        assert 'forecast_upper' in disagg.columns
        
        # Check that we get daily forecasts
        assert len(disagg) == 14  # 2 weeks of daily data
    
    def test_disaggregation_with_external_regressors(self):
        """Test disaggregation with external regressors."""
        # Remove 'dt' column from x_reg and x_future since it causes issues
        x_reg_no_dt = self.x_reg.drop('dt', axis=1)
        x_future_no_dt = self.x_future.drop('dt', axis=1)
        
        result = disaggregate_forecast(
            self.history,
            self.aggregated_history,
            self.aggregated_forecast,
            dt_units='D',
            period_agg=7,
            period_disagg=1,
            x_reg=x_reg_no_dt,
            x_future=x_future_no_dt,
            x_cols_seasonal_interactions=['marketing_spend']
        )
        
        # Check that disaggregation still works with external regressors
        disagg = result['disaggregated']
        assert len(disagg) == 14
        assert all(col in disagg.columns for col in ['dt', 'forecast', 'forecast_lower', 'forecast_upper'])
    
    def test_different_time_units(self):
        """Test disaggregation with different time units."""
        # Use a simpler case: 3-day periods aggregated to weekly
        # Create 3-day data (21 days total = 3 weeks)
        daily_dates = pd.date_range('2022-01-01', periods=21, freq='D')
        daily_actual = np.random.normal(10, 2, 21)
        daily_history = pd.DataFrame({
            'dt': daily_dates,
            'actual': daily_actual
        })
        
        # Create weekly aggregated data (3 weeks)
        weekly_dates = pd.date_range('2022-01-07', periods=3, freq='7D')
        weekly_agg = pd.DataFrame({
            'dt': weekly_dates,
            'actual': [daily_actual[:7].sum(), daily_actual[7:14].sum(), daily_actual[14:].sum()]
        })
        
        # Create weekly forecast (1 week = 7 days)
        fcst_dates = pd.date_range('2022-01-28', periods=1, freq='7D')
        weekly_fcst = pd.DataFrame({
            'dt': fcst_dates,
            'forecast': [250],
            'forecast_lower': [200],
            'forecast_upper': [300]
        })
        
        result = disaggregate_forecast(
            daily_history,
            weekly_agg,
            weekly_fcst,
            dt_units='D',
            period_agg=7,  # 7 days = 1 week
            period_disagg=1  # 1 day
        )
        
        # Should get 7 days of forecast (1 week)
        disagg = result['disaggregated']
        assert len(disagg) == 7
    
    def test_forecast_bounds(self):
        """Test that forecast bounds are reasonable."""
        result = disaggregate_forecast(
            self.history,
            self.aggregated_history,
            self.aggregated_forecast,
            dt_units='D',
            period_agg=7,
            period_disagg=1
        )
        
        disagg = result['disaggregated']
        
        # Check that lower bounds are less than or equal to forecast
        assert all(disagg['forecast_lower'] <= disagg['forecast'])
        
        # Check that upper bounds are greater than or equal to forecast
        assert all(disagg['forecast_upper'] >= disagg['forecast'])
        
        # Check that bounds are reasonable (not negative for positive forecasts)
        assert all(disagg['forecast_lower'] >= 0)
    
    def test_model_coefficients(self):
        """Test that model coefficients are returned."""
        result = disaggregate_forecast(
            self.history,
            self.aggregated_history,
            self.aggregated_forecast,
            dt_units='D',
            period_agg=7,
            period_disagg=1
        )
        
        # Check coefficients structure
        coeffs = result['coefficients']
        assert isinstance(coeffs, pd.DataFrame)
        assert 'regressor' in coeffs.columns
        
        # Check residuals structure
        residuals = result['residuals']
        assert isinstance(residuals, pd.DataFrame)
        assert 'residual' in residuals.columns
    
    def test_aggregation_consistency(self):
        """Test that disaggregated forecasts sum to aggregated forecasts."""
        result = disaggregate_forecast(
            self.history,
            self.aggregated_history,
            self.aggregated_forecast,
            dt_units='D',
            period_agg=7,
            period_disagg=1
        )
        
        disagg = result['disaggregated']
        
        # Sum daily forecasts by week
        week1_sum = disagg['forecast'][:7].sum()
        week2_sum = disagg['forecast'][7:].sum()
        
        # Should be close to aggregated forecasts (allowing for some numerical error)
        assert abs(week1_sum - 85) < 1e-6
        assert abs(week2_sum - 95) < 1e-6
    
    def test_edge_case_minimal_data(self):
        """Test with minimal data to ensure function doesn't crash."""
        # Create minimal history (7 days)
        minimal_dates = pd.date_range('2022-01-01', periods=7, freq='D')
        minimal_history = pd.DataFrame({
            'dt': minimal_dates,
            'actual': [10, 12, 8, 15, 11, 9, 13]
        })
        
        # Create single aggregated period
        agg_date = pd.date_range('2022-01-07', periods=1, freq='7D')
        minimal_agg = pd.DataFrame({
            'dt': agg_date,
            'actual': [78]
        })
        
        # Create single forecast period
        fcst_date = pd.date_range('2022-01-14', periods=1, freq='7D')
        minimal_fcst = pd.DataFrame({
            'dt': fcst_date,
            'forecast': [85],
            'forecast_lower': [75],
            'forecast_upper': [95]
        })
        
        result = disaggregate_forecast(
            minimal_history,
            minimal_agg,
            minimal_fcst,
            dt_units='D',
            period_agg=7,
            period_disagg=1
        )
        
        # Should still work with minimal data
        disagg = result['disaggregated']
        assert len(disagg) == 7
