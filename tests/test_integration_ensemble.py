import pandas as pd
import numpy as np
import pytest
from clairvoyants import Clairvoyant
from clairvoyants import ensemble

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

