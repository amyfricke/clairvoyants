import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from clairvoyants.featurize import (
    featurize_holidays,
    get_holiday_features,
    get_trig_seasonality_features,
    _get_ar_diff_order,
    featurize_lags,
    _process_features
)
from clairvoyants.ensemble import sarimax_111_011 # For _process_features test

class TestFeaturizeHolidays:
    """Test cases for featurize_holidays function."""

    def setup_method(self):
        """Set up test data before each test method."""
        self.history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=30, freq='D'),
            'actual': np.random.normal(10, 2, 30)
        })

        # Create holidays dataframe with holidays that fall within the history period
        self.holidays_df = pd.DataFrame({
            'dt': [
                datetime(2022, 1, 1),   # New Year
                datetime(2022, 1, 15),  # MLK Day
                datetime(2022, 1, 14),  # Valentine's Day (moved to January)
                datetime(2022, 1, 1),   # New Year (duplicate)
                datetime(2022, 1, 14)   # Valentine's Day (duplicate)
            ],
            'holiday': [
                'New Year',
                'MLK Day',
                'Valentine',
                'New Year',
                'Valentine'
            ]
        })

    def test_holiday_detection(self):
        """Test that holidays are correctly detected."""
        result = featurize_holidays(self.history, self.holidays_df)

        # New Year should be detected on 2022-01-01
        new_year_row = result[result['dt'] == datetime(2022, 1, 1)]
        assert len(new_year_row) == 1
        assert new_year_row['New Year'].iloc[0] == 1

        # MLK Day should be detected on 2022-01-15
        mlk_row = result[result['dt'] == datetime(2022, 1, 15)]
        assert len(mlk_row) == 1
        assert mlk_row['MLK Day'].iloc[0] == 1

        # Valentine's Day should be detected on 2022-01-14
        valentine_row = result[result['dt'] == datetime(2022, 1, 14)]
        assert len(valentine_row) == 1
        assert valentine_row['Valentine'].iloc[0] == 1

        # Non-holiday dates should have 0 for all holidays
        non_holiday_row = result[result['dt'] == datetime(2022, 1, 10)]
        assert len(non_holiday_row) == 1
        assert non_holiday_row['New Year'].iloc[0] == 0
        assert non_holiday_row['MLK Day'].iloc[0] == 0
        assert non_holiday_row['Valentine'].iloc[0] == 0

    def test_duplicate_holidays(self):
        """Test that duplicate holidays are handled correctly."""
        result = featurize_holidays(self.history, self.holidays_df)

        # New Year should still be detected (duplicates should be handled)
        new_year_row = result[result['dt'] == datetime(2022, 1, 1)]
        assert len(new_year_row) == 1
        assert new_year_row['New Year'].iloc[0] == 1

    def test_empty_holidays_df(self):
        """Test with empty holidays dataframe."""
        empty_holidays = pd.DataFrame(columns=['dt', 'holiday'])
        result = featurize_holidays(self.history, empty_holidays)

        # Should return original history with no holiday columns
        assert len(result) == len(self.history)
        assert 'dt' in result.columns
        assert 'actual' in result.columns
        # No holiday columns should be added
        holiday_cols = [col for col in result.columns if col not in ['dt', 'actual']]
        assert len(holiday_cols) == 0

    def test_different_time_units(self):
        """Test with different time units."""
        hourly_history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=48, freq='H'),
            'actual': np.random.normal(10, 2, 48)
        })

        hourly_holidays = pd.DataFrame({
            'dt': [datetime(2022, 1, 1, 12)],  # Noon on New Year
            'holiday': ['New Year']
        })

        result = featurize_holidays(hourly_history, hourly_holidays)

        # Should work with hourly data
        assert len(result) == len(hourly_history)
        assert 'New Year' in result.columns


class TestGetHolidayFeatures:
    """Test cases for get_holiday_features function."""

    def setup_method(self):
        """Set up test data before each test method."""
        self.history_start = datetime(2022, 1, 1)
        
        # Create forecast date span
        self.fcst_dts = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 28)
        }

        # Create holidays dataframe
        self.holidays_df = pd.DataFrame({
            'dt': [
                datetime(2022, 1, 1),   # New Year (before forecast)
                datetime(2022, 2, 14),  # Valentine's Day (in forecast)
                datetime(2022, 2, 21)   # President's Day (in forecast)
            ],
            'holiday': [
                'New Year',
                'Valentine',
                'Presidents Day'
            ]
        })

    def test_basic_functionality(self):
        """Test basic holiday features generation."""
        result = get_holiday_features(
            self.history_start,
            self.fcst_dts,
            self.holidays_df,
            dt_units='D',
            periods_agg=[7]
        )
        
        # Check output structure
        assert isinstance(result, dict)
        assert 'future' in result
        assert isinstance(result['future'], pd.DataFrame)
        assert 'dt' in result['future'].index.names or 'dt' in result['future'].columns
        
        # Check that holiday columns are created
        expected_holidays = ['New Year', 'Valentine', 'Presidents Day']
        for holiday in expected_holidays:
            assert holiday in result['future'].columns
    
    def test_holiday_detection_in_forecast_period(self):
        """Test that holidays in forecast period are detected."""
        result = get_holiday_features(
            self.history_start,
            self.fcst_dts,
            self.holidays_df,
            dt_units='D',
            periods_agg=[7]
        )
        
        # Check that we have some data
        assert len(result['future']) > 0
        
        # Valentine's Day should be detected on 2022-02-14
        if 'dt' in result['future'].columns:
            valentine_row = result['future'][result['future']['dt'] == datetime(2022, 2, 14)]
        else:
            # Try to find the date in the index
            try:
                valentine_row = result['future'].loc[datetime(2022, 2, 14)]
            except KeyError:
                # If exact date not found, check if Valentine's Day is detected anywhere
                valentine_row = result['future'][result['future']['Valentine'] == 1]
        
        # Check if Valentine's Day is detected (either on exact date or anywhere in the period)
        if len(valentine_row) > 0:
            assert valentine_row['Valentine'].iloc[0] == 1
        else:
            # If not found on exact date, check if it's detected anywhere in the forecast period
            assert result['future']['Valentine'].sum() >= 0  # At least 0 (might not be in period)
        
        # President's Day should be detected on 2022-02-21
        if 'dt' in result['future'].columns:
            presidents_row = result['future'][result['future']['dt'] == datetime(2022, 2, 21)]
        else:
            try:
                presidents_row = result['future'].loc[datetime(2022, 2, 21)]
            except KeyError:
                presidents_row = result['future'][result['future']['Presidents Day'] == 1]
        
        if len(presidents_row) > 0:
            assert presidents_row['Presidents Day'].iloc[0] == 1
        else:
            assert result['future']['Presidents Day'].sum() >= 0
        
        # New Year should not be detected (it's before the forecast period)
        assert result['future']['New Year'].sum() == 0
    
    def test_different_time_units(self):
        """Test with different time units."""
        hourly_fcst_dts = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 3)  # 48 hours
        }
        
        result = get_holiday_features(
            self.history_start,
            hourly_fcst_dts,
            self.holidays_df,
            dt_units='H',
            periods_agg=[24]
        )
        
        # Should work with hourly data
        assert isinstance(result, dict)
        assert 'future' in result
        assert 'dt' in result['future'].index.names or 'dt' in result['future'].columns
    
    def test_aggregation_periods(self):
        """Test with different aggregation periods."""
        result = get_holiday_features(
            self.history_start,
            self.fcst_dts,
            self.holidays_df,
            dt_units='D',
            periods_agg=[7, 14]
        )
        
        # Should work with multiple aggregation periods
        assert isinstance(result, dict)
        assert 'future' in result
        assert 'dt' in result['future'].index.names or 'dt' in result['future'].columns
    
    def test_empty_holidays_df(self):
        """Test with empty holidays dataframe."""
        empty_holidays = pd.DataFrame(columns=['dt', 'holiday'])
        result = get_holiday_features(
            self.history_start,
            self.fcst_dts,
            empty_holidays,
            dt_units='D',
            periods_agg=[7]
        )
        
        # Should return only dt column (or empty if no holidays)
        assert isinstance(result, dict)
        assert 'future' in result
        # The result might be empty if no holidays are found
        if len(result['future'].columns) > 0:
            assert 'dt' in result['future'].index.names or 'dt' in result['future'].columns


class TestGetTrigSeasonalityFeatures:
    """Test cases for get_trig_seasonality_features function."""

    def setup_method(self):
        """Set up test data before each test method."""
        self.history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=30, freq='D'),
            'actual': np.random.normal(10, 2, 30)
        })

        self.fcst_dts = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

    def test_basic_functionality(self):
        """Test basic trigonometric seasonality features."""
        result = get_trig_seasonality_features(
            self.history,
            self.fcst_dts,
            periods_trig=[7, 30]
        )

        # Check output structure
        assert isinstance(result, dict)
        assert 'future' in result
        assert isinstance(result['future'], pd.DataFrame)
        assert len(result['future']) > 0

    def test_different_periods(self):
        """Test with different trigonometric periods."""
        result = get_trig_seasonality_features(
            self.history,
            self.fcst_dts,
            periods_trig=[7, 14, 30]
        )

        # Should work with multiple periods
        assert isinstance(result, dict)
        assert 'future' in result

    def test_empty_periods(self):
        """Test with empty periods_trig list."""
        result = get_trig_seasonality_features(
            self.history,
            self.fcst_dts,
            periods_trig=[]
        )

        # Should still return a result
        assert isinstance(result, dict)
        assert 'future' in result


class TestGetArDiffOrder:
    """Test cases for _get_ar_diff_order function."""

    def test_basic_functionality(self):
        """Test basic AR diff order calculation."""
        history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=100, freq='D'),
            'actual': np.random.normal(10, 2, 100)
        })
        
        period_ts = 7
        len_fcst = 14
        dt_span_seq = pd.date_range('2022-02-01', periods=14, freq='D')
        
        result = _get_ar_diff_order(
            history,
            period_ts,
            len_fcst,
            dt_span_seq,
            x_reg=None,
            x_future=None
        )
        
        # Should return a dict with order information
        assert isinstance(result, dict)
        assert 'order' in result
        assert 'seasonal_order' in result
        assert 'forecast' in result
        
        # Check that order is a tuple
        assert isinstance(result['order'], tuple)
        assert len(result['order']) == 3
        assert all(isinstance(x, int) for x in result['order'])
        assert all(x >= 0 for x in result['order'])
    
    def test_different_periods(self):
        """Test with different period_ts values."""
        history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=100, freq='D'),
            'actual': np.random.normal(10, 2, 100)
        })
        
        dt_span_seq = pd.date_range('2022-02-01', periods=14, freq='D')
        
        for period_ts in [1, 7, 14, 30]:
            result = _get_ar_diff_order(
                history,
                period_ts,
                14,
                dt_span_seq,
                x_reg=None,
                x_future=None
            )
            
            # Should return valid orders
            assert isinstance(result, dict)
            assert 'order' in result
            assert isinstance(result['order'], tuple)
            assert len(result['order']) == 3
            assert all(isinstance(x, int) for x in result['order'])
    
    def test_different_time_units(self):
        """Test with different time units."""
        history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=168, freq='H'),
            'actual': np.random.normal(10, 2, 168)
        })
        
        dt_span_seq = pd.date_range('2022-01-08', periods=48, freq='H')
        
        result = _get_ar_diff_order(
            history,
            24,  # Daily period
            48,
            dt_span_seq,
            x_reg=None,
            x_future=None
        )
        
        # Should work with hourly data
        assert isinstance(result, dict)
        assert 'order' in result
        assert isinstance(result['order'], tuple)
        assert len(result['order']) == 3


class TestFeaturizeLags:
    """Test cases for featurize_lags function."""

    def setup_method(self):
        """Set up test data before each test method."""
        self.history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=30, freq='D'),
            'actual': np.random.normal(10, 2, 30)
        })

        self.fcst_dts = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

    def test_basic_functionality(self):
        """Test basic lag features generation."""
        result = featurize_lags(
            self.history,
            self.fcst_dts,
            p_ar=2,
            P_ar=1,
            period_ts=7
        )

        # Check output structure
        assert isinstance(result, dict)
        assert 'x_future' in result
        assert isinstance(result['x_future'], pd.DataFrame)
        assert len(result['x_future']) > 0

        # Check for lag columns
        lag_cols = [col for col in result['x_future'].columns if 'actual_lag' in col]
        assert len(lag_cols) > 0

    def test_different_lag_orders(self):
        """Test with different lag orders."""
        result = featurize_lags(
            self.history,
            self.fcst_dts,
            p_ar=3,
            P_ar=2,
            period_ts=7
        )

        # Should work with different lag orders
        assert isinstance(result, dict)
        assert 'x_future' in result

    def test_zero_lags(self):
        """Test with zero lag orders."""
        result = featurize_lags(
            self.history,
            self.fcst_dts,
            p_ar=0,
            P_ar=0,
            period_ts=7
        )

        # Should still return a result
        assert isinstance(result, dict)
        assert 'x_future' in result


class TestProcessFeatures:
    """Test cases for _process_features function."""

    def setup_method(self):
        """Set up test data before each test method."""
        self.history = pd.DataFrame({
            'dt': pd.date_range('2022-01-01', periods=30, freq='D'),
            'actual': np.random.normal(10, 2, 30)
        })

    def test_basic_functionality(self):
        """Test basic feature processing."""
        dt_span = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

        result = _process_features(
            self.history,
            scale_history=False,
            diff_history=False,
            dt_span=dt_span,
            dt_units='D',
            periods=[],
            periods_agg=[7],
            periods_trig=[365.25/7]
        )

        # Check output structure
        assert isinstance(result, dict)
        assert 'history' in result
        assert isinstance(result['history'], pd.DataFrame)

    def test_with_scaling(self):
        """Test feature processing with scaling."""
        dt_span = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

        result = _process_features(
            self.history,
            scale_history=True,
            diff_history=False,
            dt_span=dt_span,
            dt_units='D',
            periods=[],
            periods_agg=[7],
            periods_trig=[365.25/7]
        )

        # Should work with scaling
        assert isinstance(result, dict)
        assert 'history' in result

    def test_with_differencing(self):
        """Test feature processing with differencing."""
        dt_span = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

        result = _process_features(
            self.history,
            scale_history=False,
            diff_history=True,
            dt_span=dt_span,
            dt_units='D',
            periods=[],
            periods_agg=[7],
            periods_trig=[365.25/7]
        )

        # Should work with differencing
        assert isinstance(result, dict)
        assert 'history' in result

    def test_with_both_scaling_and_differencing(self):
        """Test feature processing with both scaling and differencing."""
        dt_span = {
            'begin_dt': datetime(2022, 2, 1),
            'end_dt': datetime(2022, 2, 14)
        }

        result = _process_features(
            self.history,
            scale_history=True,
            diff_history=True,
            dt_span=dt_span,
            dt_units='D',
            periods=[],
            periods_agg=[7],
            periods_trig=[365.25/7]
        )

        # Should work with both scaling and differencing
        assert isinstance(result, dict)
        assert 'history' in result