# Clairvoyant: Ensemble Forecasting and Time Series Analysis

Clairvoyant is a package for modeling and/or forecasting time series data based on an ensemble of models that may fit various levels of seasonality, irregular holiday effects, changing trends using several classes of time series models. Some of these include structural time series approaches such as Prophet or Bayesian dynamic linear models, feedforward neural networks, and various SARIMAX models.

## Other forecasting packages

- Rob Hyndman's [forecast package](http://robjhyndman.com/software/forecast/)
- [Statsmodels](http://statsmodels.sourceforge.net/)
- [Prophet](https://facebook.github.io/prophet/)

## Installation

```shell
pip install clairvoyant 
```

### Example usage

```python
  >>> from clairvoyant import Clairvoyant
  >>> e = Clairvoyant()
  >>> e.fit_ensemble(df)  # df is a pandas.DataFrame with 'actual' and 'dt' columns
  >>> e.get_out_of_time_validation(df)
  ```