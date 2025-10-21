"""
Test parallel ensemble fitting functionality.
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime
from clairvoyants import Clairvoyant
from clairvoyants import ensemble


def test_parallel_vs_sequential():
    """Test that parallel and sequential fitting produce the same results."""
    # Create synthetic time series data
    n_days = 60
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    actual = np.sin(np.linspace(0, 6 * np.pi, n_days)) + np.random.normal(0, 0.1, n_days) + 10
    df = pd.DataFrame({'dt': dates, 'actual': actual})

    # Test with parallel=True
    clair_parallel = Clairvoyant(
        periods_agg=[7], 
        periods=[7],
        periods_trig=[365.25/7], 
        transform='none',
        models=[ensemble.sarimax_013_011, ensemble.sarimax_003_001]
    )
    
    # Test with parallel=False
    clair_sequential = Clairvoyant(
        periods_agg=[7], 
        periods=[7],
        periods_trig=[365.25/7], 
        transform='none',
        models=[ensemble.sarimax_013_011, ensemble.sarimax_003_001]
    )
    
    # Fit both versions
    clair_parallel.fit_ensemble(df, parallel=True)
    clair_sequential.fit_ensemble(df, parallel=False)
    
    # Check that both produce consensus forecasts
    assert clair_parallel.forecast['consensus'] is not None
    assert clair_sequential.forecast['consensus'] is not None
    
    # Check that the consensus forecasts are similar (allowing for small numerical differences)
    parallel_forecast = clair_parallel.forecast['consensus']['forecast']
    sequential_forecast = clair_sequential.forecast['consensus']['forecast']
    
    # They should be very close (within numerical precision)
    np.testing.assert_allclose(parallel_forecast, sequential_forecast, rtol=1e-10)


def test_parallel_performance():
    """Test that parallel fitting works and doesn't crash."""
    # Create synthetic time series data
    n_days = 100
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    actual = np.sin(np.linspace(0, 6 * np.pi, n_days)) + np.random.normal(0, 0.1, n_days) + 10
    df = pd.DataFrame({'dt': dates, 'actual': actual})

    # Test with multiple models
    models = [ensemble.sarimax_013_011, ensemble.sarimax_003_001, 
              ensemble.auto_sarif_nnet, ensemble.auto_scarf_nnet]
    
    clair = Clairvoyant(
        periods_agg=[7], 
        periods=[7],
        periods_trig=[365.25/7], 
        transform='none',
        models=models
    )
    
    # Test that parallel execution works without errors
    clair.fit_ensemble(df, parallel=True)
    
    # Check that consensus forecast is produced
    assert clair.forecast['consensus'] is not None
    assert 'forecast' in clair.forecast['consensus'].columns
    
    # Test that sequential execution also works
    clair.fit_ensemble(df, parallel=False)
    assert clair.forecast['consensus'] is not None


def test_parallel_with_larger_dataset():
    """Test parallel fitting with a larger dataset where benefits are more likely."""
    # Create larger synthetic time series data
    n_days = 500
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    actual = np.sin(np.linspace(0, 6 * np.pi, n_days)) + np.random.normal(0, 0.1, n_days) + 10
    df = pd.DataFrame({'dt': dates, 'actual': actual})

    # Test with multiple models
    models = [ensemble.sarimax_013_011, ensemble.sarimax_003_001]
    
    clair = Clairvoyant(
        periods_agg=[7], 
        periods=[7],
        periods_trig=[365.25/7], 
        transform='none',
        models=models
    )
    
    # Test that parallel execution works
    clair.fit_ensemble(df, parallel=True)
    assert clair.forecast['consensus'] is not None
    assert 'forecast' in clair.forecast['consensus'].columns


def test_single_model_no_parallel():
    """Test that single model doesn't use parallel processing."""
    # Create synthetic time series data
    n_days = 30
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    actual = np.sin(np.linspace(0, 6 * np.pi, n_days)) + np.random.normal(0, 0.1, n_days) + 10
    df = pd.DataFrame({'dt': dates, 'actual': actual})

    # Test with single model
    clair = Clairvoyant(
        periods_agg=[7], 
        periods=[7],
        periods_trig=[365.25/7], 
        transform='none',
        models=[ensemble.sarimax_013_011]  # Single model
    )
    
    # Should work without issues (parallel=False path)
    clair.fit_ensemble(df, parallel=True)
    
    # Check that consensus forecast is produced
    assert clair.forecast['consensus'] is not None
    assert 'forecast' in clair.forecast['consensus'].columns
