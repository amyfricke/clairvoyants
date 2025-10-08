# Clairvoyants: Ensemble Forecasting and Time Series Analysis

Clairvoyants is a comprehensive Python package for ensemble forecasting and time series analysis. It combines multiple forecasting methodologies (SARIMAX, Prophet, neural networks) to provide robust predictions with attribution analysis and disaggregation capabilities.

## Features

- **Ensemble Forecasting**: Combines multiple models (SARIMAX, Prophet, neural networks) for robust predictions
- **Attribution Analysis**: Understand driver contributions to time series changes
- **Multi-level Aggregation**: Forecast at different time granularities and disaggregate
- **Holiday Effects**: Model irregular calendar events and seasonal patterns
- **External Regressors**: Incorporate marketing spend, economic indicators, and other features
- **Confidence Intervals**: Generate prediction intervals for uncertainty quantification

## Installation

### From GitHub (Development Version)
```bash
pip install git+https://github.com/amyfricke/clairvoyants.git
```

### From Source
```bash
git clone https://github.com/amyfricke/clairvoyants.git
cd clairvoyants
pip install -e .
```

### Dependencies
- Python >= 3.6
- pandas >= 1.0.4
- numpy >= 1.15.4
- scikit-learn >= 1.0.0
- statsmodels >= 0.11.0
- prophet >= 1.0.1
- pmdarima >= 1.8.3

## Quick Start

### Basic Forecasting

```python
import pandas as pd
import numpy as np
from datetime import datetime
from clairvoyants import Clairvoyant
from clairvoyants import ensemble

# Create sample data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
actual = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100) + 10
df = pd.DataFrame({'dt': dates, 'actual': actual})

# Initialize and fit ensemble
clair = Clairvoyant(
    periods_agg=[7], 
    periods=[7],
    periods_trig=[365.25/7], 
    transform='none',
    models=[
        ensemble.sarimax_013_011,
        ensemble.auto_sarimax,
        ensemble.auto_scarf_nnet
    ]
)

# Fit ensemble and generate forecast
clair.fit_ensemble(df)

# Get consensus forecast
forecast = clair.forecast['consensus']
print(forecast.head())
```

### With External Regressors

```python
# Add external features
features = pd.DataFrame({
    'dt': dates,
    'marketing_spend': np.random.exponential(100, 100),
    'holiday_indicator': (dates.day == 1).astype(int)
})

# Fit with features
clair.fit_ensemble(
    df=df,
    x_features=features,
    training_end_dt=datetime(2020, 12, 1),
    forecast_end_dt=datetime(2021, 3, 1)
)

# Get attribution analysis
attributions = clair.ensemble_model_artifacts['attributions']['consensus']
print(attributions.head())
```

### With Holiday Effects

```python
# Define holidays
holidays_df = pd.DataFrame({
    'dt': [datetime(2020, 12, 25), datetime(2021, 1, 1)],
    'holiday': ['Christmas', 'New Year']
})

# Configure with holidays
clair = Clairvoyant(
    models=[ensemble.sarimax_013_011, ensemble.prophet_linear_lt],
    holidays_df=holidays_df,
    periods_trig=[365.25]  # Annual seasonality
)

clair.fit_ensemble(df)
```

## API Reference

### Clairvoyant Class

The main forecasting class that orchestrates ensemble modeling.

#### Parameters

- `dt_units` (str): Time series frequency ('D', 'H', 'W', 'MS')
- `periods_agg` (list): Periods to aggregate over (e.g., [7] for weekly)
- `transform` (str): Transformation ('log' or 'none')
- `periods` (list): Seasonal periods for models
- `periods_trig` (list): Trigonometric seasonal periods
- `models` (list): Ensemble models to use
- `holidays_df` (DataFrame): Holiday calendar
- `consensus_method` (function): Method for consensus (np.median, np.mean)
- `pred_level` (float): Prediction interval level (0.8, 0.9, etc.)

#### Methods

- `fit_ensemble(df, x_features=None, training_begin_dt=None, training_end_dt=None, forecast_end_dt=None)`: Fit ensemble models
- `disaggregate_forecasts(x_features_col_subset=None)`: Disaggregate to original granularity
- `get_out_of_time_validation(df)`: Calculate validation metrics

### Available Models

- `ensemble.sarimax_013_011`: SARIMAX(0,1,3)(0,1,1)
- `ensemble.sarimax_003_001`: SARIMAX(0,0,3)(0,0,1)
- `ensemble.auto_sarimax`: Auto-selected SARIMAX
- `ensemble.prophet_linear_lt`: Prophet with linear trend
- `ensemble.auto_sarif_nnet`: Auto-regressive neural network
- `ensemble.auto_scarf_nnet`: Scaled auto-regressive neural network

## Examples

### Attribution Analysis

```python
# Load example data (see notebooks for full examples)
tutoring_subscribers = pd.read_csv("examples/tutoring_subscribers.csv")
tutoring_features = pd.read_csv("examples/tutoring_covariates.csv")

# Configure for attribution
clair = Clairvoyant(
    models=[ensemble.sarimax_013_011, ensemble.auto_sarif_nnet],
    transform='none',
    holidays_df=holidays_df
)

# Fit with features
clair.fit_ensemble(
    df=tutoring_subscribers,
    x_features=tutoring_features
)

# Get attribution results
attributions = clair.ensemble_model_artifacts['attributions']['consensus']
```

### Multi-level Forecasting

```python
# Forecast at weekly level, then disaggregate to daily
clair = Clairvoyant(
    periods_agg=[7],  # Weekly aggregation
    models=[ensemble.auto_sarimax, ensemble.prophet_linear_lt]
)

# Fit ensemble
clair.fit_ensemble(df, x_features=features)

# Disaggregate to daily
clair.disaggregate_forecasts(x_features_col_subset=['marketing_spend'])

# Get daily forecast
daily_forecast = clair.forecast['disaggregated']['period1']
```

## Data Requirements

### Input DataFrame
Your time series data must have:
- `dt`: datetime column (pandas datetime)
- `actual`: target variable (numeric)

### External Features (Optional)
If using external regressors:
- `dt`: datetime column matching your target data
- Additional columns for each feature
- Must cover both training and forecast periods

### Holiday Data (Optional)
For holiday effects:
- `dt`: datetime column
- `holiday`: holiday name (string)

## Troubleshooting

### Common Issues

**ImportError: No module named 'clairvoyants'**
```bash
pip install -e .  # If installing from source
```

**TypeError: __init__() got multiple values for argument 'schema'**
- This is a pandasql compatibility issue (now fixed)
- Restart your Jupyter kernel and reinstall the package

**ConvergenceWarning: Maximum iterations reached**
- Neural networks may not converge with limited data
- Try reducing model complexity or increasing training data

**ValueError: Missing required columns**
- Ensure your DataFrame has 'dt' and 'actual' columns
- Check that datetime column is properly formatted

### Performance Tips

- **Large datasets**: Use fewer models or reduce training window
- **Slow convergence**: Increase `max_iter` in neural network models
- **Memory issues**: Process data in chunks or reduce feature set

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Clairvoyants in your research, please cite:

```bibtex
@software{clairvoyants,
  title={Clairvoyants: Ensemble Forecasting and Time Series Analysis},
  author={Amy Richardson Fricke},
  year={2024},
  url={https://github.com/amyfricke/clairvoyants}
}
```

## Related Packages

- [Prophet](https://facebook.github.io/prophet/) - Facebook's forecasting tool
- [Statsmodels](https://www.statsmodels.org/) - Statistical modeling
- [pmdarima](https://pypi.org/project/pmdarima/) - Auto-ARIMA
- [scikit-learn](https://scikit-learn.org/) - Machine learning

## Support

- **Issues**: [GitHub Issues](https://github.com/amyfricke/clairvoyants/issues)
- **Documentation**: See `notebooks/` folder for examples
- **Email**: amy.r.fricke@gmail.com