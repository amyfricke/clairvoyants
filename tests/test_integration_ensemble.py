from datetime import datetime
import pandas as pd
import numpy as np
import pytest
from clairvoyants import Clairvoyant
from clairvoyants import ensemble
from .test_utils import get_example_file_path

def test_ensemble_forecasting_basic():
    # Create synthetic time series data
    n_days = 60
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    actual = np.sin(np.linspace(0, 6 * np.pi, n_days)) + np.random.normal(0, 0.1, n_days) + 10
    df = pd.DataFrame({'dt': dates, 'actual': actual})

    # Instantiate and fit the ensemble forecaster
    clair = Clairvoyant(periods_agg=[7], periods=[7],
     periods_trig=[365.25/7], transform='none',
     models=[ensemble.sarimax_013_011,
             ensemble.auto_sarif_nnet,
             ensemble.auto_scarf_nnet,
             ensemble.sarimax_003_001])
    clair.fit_ensemble(df)

    # Check that consensus forecast is produced
    consensus = clair.forecast['consensus']
    assert consensus is not None
    assert 'forecast' in consensus.columns
    assert 'forecast_lower' in consensus.columns
    assert 'forecast_upper' in consensus.columns
    assert len(consensus) > 0

def test_ensemble_forecasting_moderate():
    # Read example data
    tutoring_subscribers = pd.read_csv(get_example_file_path("tutoring_subscribers.csv"))
    tutoring_subscribers['dt'] = pd.to_datetime(tutoring_subscribers.dt)

    tutoring_features = pd.read_csv(get_example_file_path("tutoring_covariates.csv"))
    tutoring_features['dt'] = pd.to_datetime(tutoring_features.dt)

    holidays_df = pd.read_csv(get_example_file_path("holidays_df_example.csv"))
    holidays_df['dt'] = pd.to_datetime(holidays_df.dt)

    # Instantiate and fit the ensemble forecaster
    clair = Clairvoyant(models=[ensemble.sarimax_013_011,
      ensemble.auto_sarif_nnet, ensemble.auto_scarf_nnet,
      ensemble.sarimax_003_001], transform='none',
      periods_trig=[], holidays_df=holidays_df)
    clair.fit_ensemble(df=tutoring_subscribers, x_features=tutoring_features,
    training_end_dt=datetime(2022, 1, 10),vforecast_end_dt=datetime(2022, 9, 14),)

    # Check that consensus forecast is produced
    consensus = clair.forecast['consensus']
    assert consensus is not None
    assert 'forecast' in consensus.columns
    assert 'forecast_lower' in consensus.columns
    assert 'forecast_upper' in consensus.columns
    assert len(consensus) > 0
