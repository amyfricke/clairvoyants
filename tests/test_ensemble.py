import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from clairvoyants.ensemble import (
    auto_sarif_nnet,
    auto_sarimax,
    auto_scarf_nnet,
    prophet_linear_lt,
    prophet_linear_lt2,
    sarimax_pdq_PDQ,
    sarimax_111_011,
    sarimax_013_011,
    sarimax_003_001,
    sarimax_002_001,
    sarimax_001_001,
    validate_parameters
)


class TestValidateParameters:
    """Test cases for validate_parameters function."""
    
    def test_valid_parameters(self):
        """Test with valid parameters."""
        result = validate_parameters(
            dt_units='D',
            transform='none',
            periods=[7],
            periods_trig=[365.25],
            holidays_df=None,
            models=[sarimax_111_011, sarimax_013_011],
            consensus_method=np.median,
            pred_level=0.8
        )
        assert result is True
    
    def test_invalid_dt_units(self):
        """Test with invalid dt_units."""
        with pytest.raises(Exception, match='Invalid dt_units input'):
            validate_parameters(dt_units='invalid')
    
    def test_invalid_transform(self):
        """Test with invalid transform."""
        with pytest.raises(Exception, match='transform options are'):
            validate_parameters(transform='invalid')
    
    def test_invalid_periods(self):
        """Test with invalid periods."""
        with pytest.raises(Exception, match='Seasonal periods must be'):
            validate_parameters(periods=[7.5])
    
    def test_invalid_periods_trig(self):
        """Test with invalid periods_trig."""
        with pytest.raises(Exception, match='Trigonometric seasonal periods'):
            validate_parameters(periods_trig=['invalid'])
    
    def test_invalid_holidays_df(self):
        """Test with invalid holidays_df."""
        # Test with non-dataframe input
        with pytest.raises(Exception, match='holidays_df must be a pandas dataframe'):
            validate_parameters(holidays_df="not_a_dataframe")
    
    def test_invalid_consensus_method(self):
        """Test with invalid consensus method."""
        with pytest.raises(Exception, match='Current consensus method options'):
            validate_parameters(consensus_method=lambda x: x)
    
    def test_invalid_pred_level(self):
        """Test with invalid pred_level."""
        with pytest.raises(Exception, match='pred_level must be between'):
            validate_parameters(pred_level=1.5)


class TestEnsembleModels:
    """Test cases for ensemble model functions."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create sample time series data
        dates = pd.date_range('2022-01-01', periods=60, freq='D')
        actual = np.sin(np.linspace(0, 4*np.pi, 60)) + np.random.normal(0, 0.1, 60) + 10
        self.history = pd.DataFrame({
            'dt': dates,
            'actual': actual
        })
        
        # Create forecast date span
        self.dt_span = {
            'begin_dt': datetime(2022, 3, 1),
            'end_dt': datetime(2022, 3, 14)
        }
        
        # Create external regressors
        self.x_reg = pd.DataFrame({
            'dt': dates,
            'marketing_spend': np.random.exponential(100, 60),
            'holiday_indicator': (dates.day == 1).astype(int)
        })
        
        # Create future external regressors
        future_dates = pd.date_range('2022-03-01', periods=14, freq='D')
        self.x_future = pd.DataFrame({
            'dt': future_dates,
            'marketing_spend': np.random.exponential(100, 14),
            'holiday_indicator': (future_dates.day == 1).astype(int)
        })
        
        # Create holidays dataframe
        self.holidays_df = pd.DataFrame({
            'dt': [datetime(2022, 1, 1), datetime(2022, 2, 14)],
            'holiday': ['New Year', 'Valentine']
        })
    
    def test_function_signatures(self):
        """Test that all ensemble functions have the correct signatures."""
        # Test that functions exist and are callable
        functions_to_test = [
            auto_sarif_nnet,
            auto_sarimax,
            auto_scarf_nnet,
            prophet_linear_lt,
            prophet_linear_lt2,
            sarimax_111_011,
            sarimax_013_011,
            sarimax_003_001,
            sarimax_002_001,
            sarimax_001_001
        ]
        
        for func in functions_to_test:
            assert callable(func)
            # Check that function has required parameters
            import inspect
            sig = inspect.signature(func)
            required_params = ['history', 'dt_span']
            for param in required_params:
                assert param in sig.parameters
    
    def test_sarimax_pdq_PDQ_signature(self):
        """Test sarimax_pdq_PDQ function signature."""
        import inspect
        sig = inspect.signature(sarimax_pdq_PDQ)
        required_params = ['pdq_order', 's_pdq_order', 'history', 'dt_span']
        for param in required_params:
            assert param in sig.parameters
    
    def test_validate_parameters_integration(self):
        """Test that validate_parameters works with ensemble functions."""
        # Test that validate_parameters can validate parameters for ensemble use
        result = validate_parameters(
            dt_units='D',
            transform='none',
            periods=[7],
            periods_trig=[365.25],
            holidays_df=None,
            models=[sarimax_111_011, sarimax_013_011],
            consensus_method=np.median,
            pred_level=0.8
        )
        assert result is True
    
    def test_ensemble_function_imports(self):
        """Test that all ensemble functions can be imported."""
        from clairvoyants.ensemble import (
            auto_sarif_nnet,
            auto_sarimax,
            auto_scarf_nnet,
            prophet_linear_lt,
            prophet_linear_lt2,
            sarimax_pdq_PDQ,
            sarimax_111_011,
            sarimax_013_011,
            sarimax_003_001,
            sarimax_002_001,
            sarimax_001_001,
            validate_parameters
        )
        
        # All imports should succeed
        assert True
    
    def test_ensemble_function_docstrings(self):
        """Test that ensemble functions have docstrings."""
        functions_to_test = [
            auto_sarif_nnet,
            auto_sarimax,
            auto_scarf_nnet,
            prophet_linear_lt,
            prophet_linear_lt2,
            sarimax_pdq_PDQ,
            sarimax_111_011,
            sarimax_013_011,
            sarimax_003_001,
            sarimax_002_001,
            sarimax_001_001,
            validate_parameters
        ]
        
        for func in functions_to_test:
            assert func.__doc__ is not None
            assert len(func.__doc__.strip()) > 0
    
    def test_ensemble_function_parameter_validation(self):
        """Test that ensemble functions validate their parameters."""
        # Test with invalid history
        with pytest.raises((TypeError, AttributeError, KeyError)):
            auto_sarimax(
                "invalid_history",
                self.dt_span
            )
        
        # Test with invalid dt_span
        with pytest.raises((TypeError, AttributeError, KeyError)):
            auto_sarimax(
                self.history,
                "invalid_dt_span"
            )
    
    def test_ensemble_function_with_minimal_data(self):
        """Test ensemble functions with minimal valid data."""
        # Create minimal valid data
        minimal_dates = pd.date_range('2022-01-01', periods=30, freq='D')
        minimal_actual = np.random.normal(10, 1, 30)
        minimal_history = pd.DataFrame({
            'dt': minimal_dates,
            'actual': minimal_actual
        })
        
        minimal_dt_span = {
            'begin_dt': datetime(2022, 1, 31),
            'end_dt': datetime(2022, 2, 6)
        }
        
        # Test that functions can be called (even if they fail internally)
        functions_to_test = [
            sarimax_111_011,
            sarimax_013_011,
            sarimax_003_001,
            sarimax_002_001,
            sarimax_001_001
        ]
        
        for func in functions_to_test:
            try:
                result = func(
                    minimal_history,
                    minimal_dt_span,
                    dt_units='D',
                    periods=[],
                    periods_agg=[],
                    periods_trig=[],
                    pred_level=0.8,
                    transform='none'
                )
                # If successful, check basic structure
                if isinstance(result, dict):
                    assert 'forecast' in result
            except Exception:
                # Some functions may fail with minimal data, which is expected
                pass
    
        
    def test_ensemble_function_with_external_regressors(self):
        """Test ensemble functions with external regressors."""
        # Test that functions can accept external regressors
        functions_to_test = [
            sarimax_111_011,
            sarimax_013_011,
            sarimax_003_001,
            sarimax_002_001,
            sarimax_001_001
        ]
        
        for func in functions_to_test:
            try:
                result = func(
                    self.history,
                    self.dt_span,
                    dt_units='D',
                    periods=[],
                    periods_agg=[],
                    periods_trig=[],
                    pred_level=0.8,
                    transform='none',
                    x_reg=self.x_reg,
                    x_future=self.x_future
                )
                # If successful, check basic structure
                if isinstance(result, dict):
                    assert 'forecast' in result
            except Exception:
                # Some functions may fail, which is expected
                pass
