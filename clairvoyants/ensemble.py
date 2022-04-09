# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from clairvoyants.featurize import _process_features
from clairvoyants.utilities import _back_transform_df, _datetime_delta, _rescale_forecast, _int_forecast

from inspect import getfullargspec
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from prophet.utilities import regressor_coefficients
import random
from re import search
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA


def auto_sarif_nnet(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get an autoregressive feedforward neural network model
   
  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features (n/a for this model)
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
   transformed and backtransformed forecast.""" 
   
  model_dict= _process_features(
      history, scale_history=False, diff_history=True,
      dt_span=dt_span, dt_units=dt_units, 
      periods=periods, periods_agg=periods_agg, periods_trig=periods_trig,
      holidays_df=holidays_df, x_reg=x_reg, x_future=x_future, scale_x=True,
      auto_create_lags=True) 
  
  x_reg = model_dict['x_reg'].copy()
  history = model_dict['history'].copy()
  
  autoreg_ff_nnet_reg = MLPRegressor(
        hidden_layer_sizes=(x_reg.shape[1] * 2), max_iter=2000, 
        warm_start=True, random_state=1) 

  autoreg_ff_nnet_fit = autoreg_ff_nnet_reg.fit(x_reg, history.actual_diff)
 
  autoreg_ff_nnet_fcst_df = pd.DataFrame(
      {'dt':model_dict['dt_span_seq'],
       'forecast':autoreg_ff_nnet_fit.predict(model_dict['x_future'])})
  
  autoreg_ff_nnet_reg = MLPRegressor(
        hidden_layer_sizes=(x_reg.shape[1] * 2), max_iter=1000, 
        warm_start=True)
 
  autoreg_ff_nnet_std_df = pd.DataFrame()  

  for i in list(range(100)):
    autoreg_ff_nnet_fit = autoreg_ff_nnet_reg.fit(x_reg, history.actual_diff)
    
    autoreg_ff_nnet_fcst = pd.DataFrame(
        {'dt':model_dict['dt_span_seq'],
         'forecast':autoreg_ff_nnet_fit.predict(model_dict['x_future']),
         'ensemble_num':i})
    autoreg_ff_nnet_std_df = pd.concat([autoreg_ff_nnet_std_df,
                                        autoreg_ff_nnet_fcst], 
                                        ignore_index=True)    
      
  autoreg_ff_nnet_std_df = autoreg_ff_nnet_std_df.groupby('dt').aggregate(
      {'forecast':np.std})
  autoreg_ff_nnet_std_df.index = autoreg_ff_nnet_fcst_df.index
  
  z_mult = -1 * norm.ppf((1 - pred_level) / 2) 
  autoreg_ff_nnet_std_df['interval'] = z_mult * autoreg_ff_nnet_std_df.forecast
  autoreg_ff_nnet_fcst_df['forecast_lower'] = (
      autoreg_ff_nnet_fcst_df['forecast'] - autoreg_ff_nnet_std_df['interval'])
  autoreg_ff_nnet_fcst_df['forecast_upper'] = (
      autoreg_ff_nnet_fcst_df['forecast'] + autoreg_ff_nnet_std_df['interval'])
  
  autoreg_ff_nnet_fcst_df = _int_forecast(autoreg_ff_nnet_fcst_df, history,
                                          model_dict['pdq_order'][2])

  autoreg_ff_nnet_fcst_df.reset_index(inplace=True)
 
  autoreg_ff_nnet_fcst_bcktrans = _back_transform_df(autoreg_ff_nnet_fcst_df,
                                                     transform)
  
  model_residuals = pd.DataFrame({'dt':model_dict['history'].dt,
                                  'predicted':autoreg_ff_nnet_fit.predict(
                                      x_reg),
                                  'residual':(
                                      history.actual_diff -
                                      autoreg_ff_nnet_fit.predict(
                                          x_reg))})
  model_residuals = _int_forecast(model_residuals, history,
                                  model_dict['pdq_order'][2],
                                  cols_int=['predicted'])
  
                                   
  model_attributions = pd.DataFrame()
  for x_col in x_reg.columns:
    x_reg_x_col = x_reg.copy()
    x_reg_x_col[x_col] = 0
    autoreg_ff_nnet_att = pd.DataFrame()
    autoreg_ff_nnet_att['pred'] = autoreg_ff_nnet_reg.predict(x_reg_x_col)
    autoreg_ff_nnet_att = _int_forecast(autoreg_ff_nnet_att, history,
                                        model_dict['pdq_order'][2],
                                        cols_int=['pred'])

    model_attributions[x_col] = (
        model_residuals.reset_index(drop=True).predicted 
        - autoreg_ff_nnet_att.reset_index(drop=True).pred)

  
  return {'forecast':autoreg_ff_nnet_fcst_bcktrans, 
          'transformed_forecast':autoreg_ff_nnet_fcst_df,
          'coefficients':None,
          'summary':None,
          'residuals':model_residuals,
          'attributions':None}


def auto_sarimax(history,
                 dt_span,
                 dt_units='D',
                 periods=[1],
                 periods_agg=[7],
                 periods_trig=[365.25 / 7],
                 pred_level=0.8,
                 transform='none',
                 x_reg=None,
                 x_future=None,
                 holidays_df=None,
                 x_cols_seasonal_interactions=[]):
  """Function to get a SARIMAX model with order pdq, and seasonal order PDQ
   found automatically.
   
  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
   backtransformed forecast.""" 
   
  model_dict = _process_features(
      history, scale_history=False, diff_history=False,
      dt_span=dt_span, dt_units=dt_units, 
      periods=periods, periods_agg=periods_agg, periods_trig=periods_trig,
      holidays_df=holidays_df, x_reg=x_reg, x_future=x_future,
      x_cols_seasonal_interactions=x_cols_seasonal_interactions) 
    
  
  auto_sarimax_model = auto_arima(
      model_dict['history'].actual, X=model_dict['x_reg'].to_numpy(),
      max_P=1, max_Q=1, max_p=3, max_q=3, max_d=2, max_order=5, d=1,
      m=model_dict['period_ts'], method='bfgs', out_of_sample_size=1,
      stepwise=False, with_intercept=True, n_iter=20)

  model_summary = auto_sarimax_model.summary()
  model_residuals = pd.DataFrame({'dt':model_dict['history'].dt,
                                  'residual': auto_sarimax_model.resid()})

  model_coefficients = pd.read_html(model_summary.tables[1].as_html(), 
                                    header=0, index_col=0)[0]
  
  n_col_x = len(model_dict['x_reg'].columns)
  model_coefficients['regressor'] = (
      [model_coefficients.index[0]] + list(model_dict['x_reg'].columns) +
       list(model_coefficients.index[n_col_x + 1:]))
  
  model_coefficients = model_coefficients.reset_index()
  
  model_attributions = pd.DataFrame()
  model_attributions['dt'] = model_dict['history'].dt
  model_attributions['total'] = (
      model_dict['history'].actual - model_residuals.residual)
  model_attributions['actual'] = model_dict['history'].actual
  
  for x_col in model_dict['x_reg'].columns:
    x_col_mask = (model_coefficients.regressor == x_col)
    x_coef = float(model_coefficients.coef.loc[x_col_mask])
    x_var = float(model_coefficients.loc[x_col_mask, 'std err']) ** 2
    model_attributions[x_col] = x_coef * model_dict['x_reg'][x_col]
    model_attributions[x_col + '_var'] = (
        x_var * model_dict['x_reg'][x_col] ** 2)

  
  auto_sarimax_fcst = auto_sarimax_model.predict(
     model_dict['len_fcst'], X=model_dict['x_future'], return_conf_int=True,
      alpha=(1-pred_level))
  auto_sarimax_fcst_df = pd.DataFrame(
      {'dt':model_dict['dt_span_seq'],
       'forecast':auto_sarimax_fcst[0][:model_dict['len_fcst']],
       'forecast_lower':auto_sarimax_fcst[1][:model_dict['len_fcst'], 0],
       'forecast_upper':auto_sarimax_fcst[1][:model_dict['len_fcst'], 1]})
  

  auto_sarimax_fcst_bcktrans = _back_transform_df(
      auto_sarimax_fcst_df, transform)
  
  return {'forecast':auto_sarimax_fcst_bcktrans, 
          'transformed_forecast':auto_sarimax_fcst_df,
          'coefficients':model_coefficients,
          'summary':model_summary,
          'residuals':model_residuals,
          'attributions':model_attributions}


def auto_scarf_nnet(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get an autoregressive feedforward neural network model
   
  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features (n/a for this model)
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
   transformed and backtransformed forecast.""" 
   
  model_dict= _process_features(
      history, scale_history=True, diff_history=False,
      dt_span=dt_span, dt_units=dt_units, 
      periods=periods, periods_agg=periods_agg, periods_trig=periods_trig,
      holidays_df=holidays_df, x_reg=x_reg, x_future=x_future, scale_x=True,
      auto_create_lags=True) 
  
  x_reg = model_dict['x_reg'].copy()
  history = model_dict['history'].copy()
  
  autoreg_ff_nnet_reg = MLPRegressor(
        hidden_layer_sizes=(x_reg.shape[1] * 2), max_iter=2000, 
        warm_start=True, random_state=1) 

  autoreg_ff_nnet_fit = autoreg_ff_nnet_reg.fit(x_reg, history.actual_scaled)
 
  autoreg_ff_nnet_fcst_df = pd.DataFrame(
      {'dt':model_dict['dt_span_seq'],
       'forecast':autoreg_ff_nnet_fit.predict(model_dict['x_future'])})
  
  autoreg_ff_nnet_reg = MLPRegressor(
        hidden_layer_sizes=(x_reg.shape[1] * 2), max_iter=1000, 
        warm_start=True)
 
  autoreg_ff_nnet_std_df = pd.DataFrame()  

  for i in list(range(100)):
    autoreg_ff_nnet_fit = autoreg_ff_nnet_reg.fit(x_reg, history.actual_scaled)
    
    autoreg_ff_nnet_fcst = pd.DataFrame(
        {'dt':model_dict['dt_span_seq'],
         'forecast':autoreg_ff_nnet_fit.predict(model_dict['x_future']),
         'ensemble_num':i})
    autoreg_ff_nnet_std_df = pd.concat([autoreg_ff_nnet_std_df,
                                        autoreg_ff_nnet_fcst], 
                                        ignore_index=True)    
      
  autoreg_ff_nnet_std_df = autoreg_ff_nnet_std_df.groupby('dt').aggregate(
      {'forecast':np.std})
  autoreg_ff_nnet_std_df.index = autoreg_ff_nnet_fcst_df.index
  
  z_mult = -1 * norm.ppf((1 - pred_level) / 2) 
  autoreg_ff_nnet_std_df['interval'] = z_mult * autoreg_ff_nnet_std_df.forecast
  autoreg_ff_nnet_fcst_df['forecast_lower'] = (
      autoreg_ff_nnet_fcst_df['forecast'] - autoreg_ff_nnet_std_df['interval'])
  autoreg_ff_nnet_fcst_df['forecast_upper'] = (
      autoreg_ff_nnet_fcst_df['forecast'] + autoreg_ff_nnet_std_df['interval'])
  
  
  autoreg_ff_nnet_fcst_df = _rescale_forecast(autoreg_ff_nnet_fcst_df, history)

  autoreg_ff_nnet_fcst_df.reset_index(inplace=True)
 
  autoreg_ff_nnet_fcst_bcktrans = _back_transform_df(autoreg_ff_nnet_fcst_df,
                                                     transform)
  
  model_residuals = pd.DataFrame({'dt':model_dict['history'].dt,
                                  'predicted':autoreg_ff_nnet_fit.predict(
                                      x_reg),
                                  'residual':(
                                      history.actual_scaled -
                                      autoreg_ff_nnet_fit.predict(
                                          x_reg))})
  model_residuals = _rescale_forecast(model_residuals, history,
                                      cols_rescale=['predicted'])
  
                                   
  model_attributions = pd.DataFrame()
  for x_col in x_reg.columns:
    x_reg_x_col = x_reg.copy()
    x_reg_x_col[x_col] = 0
    autoreg_ff_nnet_att = pd.DataFrame()
    autoreg_ff_nnet_att['pred'] = autoreg_ff_nnet_reg.predict(x_reg_x_col)
    autoreg_ff_nnet_att = _rescale_forecast(autoreg_ff_nnet_att, history,
                                            cols_rescale=['pred'])

    model_attributions[x_col] = (
        model_residuals.reset_index(drop=True).predicted 
        - autoreg_ff_nnet_att.reset_index(drop=True).pred)
  
  return {'forecast':autoreg_ff_nnet_fcst_bcktrans, 
          'transformed_forecast':autoreg_ff_nnet_fcst_df,
          'coefficients':None,
          'summary':None,
          'residuals':model_residuals,
          'attributions':None}


def prophet_linear_lt(history,
                      dt_span,
                      dt_units='D',
                      periods=[1],
                      periods_agg=[7],
                      periods_trig=[365.25 / 7],
                      pred_level=0.8,
                      transform='none',
                      x_reg=None,
                      x_future=None,
                      holidays_df=None,
                      x_cols_seasonal_interactions=[]):
  """Function to get prophet model results with linear growth and dampened
   changepoint sensitivity (better for long-term forecasts).
   
  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features (n/a for this model)
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
   backtransformed forecast."""
  fcst_interval = int(max(periods_agg + [1]))
  dt_span_seq = pd.to_datetime(pd.bdate_range(
      str(dt_span['begin_dt'] + _datetime_delta(fcst_interval - 1, dt_units)),
      str(dt_span['end_dt']),
      freq=str(fcst_interval) + dt_units))
  
  all_periods = periods + [period * fcst_interval for period in periods_trig]
  history = history.rename(columns={'dt':'ds', 'actual':'y'})
  history = history.reset_index()
  prophet_fcst_df = pd.DataFrame({'ds':dt_span_seq})
  
  if holidays_df is not None:
    holidays_df = holidays_df[['dt', 'holiday']]
    holidays_df = holidays_df.rename(columns={'dt':'ds'})
  
  prophet_model = Prophet(
      growth='linear',
      yearly_seasonality=any([period in [364, 365, 365.25] 
                              for period in all_periods]) * 1,
      weekly_seasonality=any([period == 7
                              for period in all_periods]) * 1,
      holidays=holidays_df,
      changepoint_prior_scale=0.05,
      mcmc_samples=300,
      n_changepoints=int(np.floor(history.shape[0] / 100)),
      interval_width=pred_level)
  
  periods_auto =  [1, 7, 12, 24, 364, 365, 365.25]
  
  if any([period not in periods_auto 
          for period in all_periods]):
    for period in [period not in periods_auto
                   for period in all_periods]:
      prophet_model.add_seasonality('period' + str(period), 
                                    period, fourier_order=2)

  if x_reg is not None and x_future is not None:
    for x_reg_col in x_reg.columns:
      history[x_reg_col] = x_reg.reset_index()[x_reg_col]
      prophet_fcst_df[x_reg_col] = x_future.reset_index()[x_reg_col]
      prophet_model.add_regressor(x_reg_col)
  
  random.seed(12345)
  prophet_fit = prophet_model.fit(history, n_jobs=1)
  
  prophet_fcst = prophet_fit.predict(prophet_fcst_df)
  
  prophet_fcst_df = pd.DataFrame({'dt':dt_span_seq, 
                                  'forecast':prophet_fcst['yhat'],
                                  'forecast_lower':prophet_fcst['yhat_lower'],
                                  'forecast_upper':prophet_fcst['yhat_upper']})
  
  prophet_fitted = prophet_fit.predict(history)
  
  model_residuals = pd.DataFrame({'dt':history['ds'],
                                  'residual':history['y'] - 
                                    prophet_fitted['yhat']})
  
  model_coefficients = regressor_coefficients(prophet_fit)
  model_coefficients = model_coefficients.append(
      {'regressor': 'linear_trend', 'regressor_mode':'additive', 'center': 0,
       'coef_lower': prophet_fcst['trend_lower'][1]
       - prophet_fcst['trend_lower'][0], 
       'coef': prophet_fcst['trend'][1] - prophet_fcst['trend'][0], 
       'coef_upper':prophet_fcst['trend_upper'][1]
       - prophet_fcst['trend_upper'][0]}, ignore_index = True)
  
  if holidays_df is not None:
    for hday in holidays_df['holiday']:
      ds_hday = holidays_df['ds'].loc[holidays_df.holiday == hday]
      ds_hday = [any(ds == ds_h for ds_h in ds_hday) 
                 for ds in prophet_fitted.ds]
      model_coefficients = model_coefficients.append(
          {'regressor':hday, 'regressor_mode':'additive', 'center':0,
           'coef_lower':np.mean(prophet_fitted[hday + '_lower'][ds_hday]),
           'coef':np.mean(prophet_fitted[hday][ds_hday]),
           'coef_upper':np.mean(prophet_fitted[hday + '_upper'][ds_hday])},
          ignore_index=True)
  
  seasonality_dict = prophet_fit.seasonalities
  if len(seasonality_dict) > 0:
    i = 0
    trig_term = 'sin'
    for seas_name in seasonality_dict:
      for fo in list(range(
              1, seasonality_dict[seas_name]['fourier_order'] + 1)):
        for j in list(range(2)):
          regressor = seas_name + '_' + trig_term + '_order_' + str(fo)
          beta = prophet_fit.params['beta'][:, i]
          coef = beta / prophet_fit.y_scale
          coef_mean = np.mean(coef)
          coef_bounds = np.quantile(coef, q=[(1 - pred_level) / 2,
                                           1 - (1 - pred_level) / 2])
          model_coefficients = model_coefficients.append(
              {'regressor':regressor, 'regressor_mode':'additive', 'center':0,
               'coef_lower':coef_bounds[0], 'coef':coef_mean,
               'coef_upper':coef_bounds[1]}, ignore_index=True)
          i += 1
          trig_term = (i % 2 == 0) * 'sin' + (i % 2 == 1) * 'cos'
          
  pi_z = 1 - (1 - pred_level) / 2
  model_coefficients['se'] = (model_coefficients.coef_upper 
                              - model_coefficients.coef_lower) / norm.ppf(pi_z)
  model_attributions = pd.DataFrame()
  model_attributions['dt'] = history.ds
  model_attributions['total'] = (
      history.y - model_residuals.residual)
  model_attributions['actual'] = history.y
  
  for x_col in x_reg.columns:
    x_col_mask = (model_coefficients.regressor == x_col)
    x_coef = float(model_coefficients.coef.loc[x_col_mask])
    x_se = float(model_coefficients.se.loc[x_col_mask])
    model_attributions[x_col] = x_coef * history[x_col]
    model_attributions[x_col + '_var'] = x_se ** 2 * history[x_col] ** 2

    
  prophet_fcst_bcktrans = _back_transform_df(prophet_fcst_df, transform)
  
  return {'forecast':prophet_fcst_bcktrans, 
          'transformed_forecast':prophet_fcst_df,
          'coefficients':model_coefficients,
          'summary':None,
          'residuals':model_residuals,
          'attributions':model_attributions}

def sarimax_pdq_PDQ(pdq_order,
                    s_pdq_order,
                    history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to set a SARIMAX model of order pdq, and seasonal order PDQ
    framework.
  
  Parameters
  ----------
  pdq_order: order of the arima model to be fit
  s_pdq_order: seasonal order of the arima model to be fit (if arima
  seasonality is specified - when periods contains integers > 1
  history: pd dataframe containing the training history of the time series
    being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
    methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  transform: transformation ('log' or 'none') applied to the timeseries being
   modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
    backtransformed forecast.""" 
    
  model_dict= _process_features(
      history, scale_history=False, diff_history=False,
      dt_span=dt_span, dt_units=dt_units, 
      periods=periods, periods_agg=periods_agg, periods_trig=periods_trig,
      holidays_df=holidays_df, x_reg=x_reg, x_future=x_future,
      x_cols_seasonal_interactions=x_cols_seasonal_interactions)
   
  if len(periods) > 0:
    periods = [p for p in periods if p > 1]

  if len(periods) > 0:
    seasonal_order = s_pdq_order + (model_dict['period_ts'], )

  else:
    seasonal_order = (0, 0, 0, 0)
    
   
  sarimax_model = ARIMA(model_dict['history'].actual, exog=model_dict['x_reg'],
                        order=pdq_order, seasonal_order=seasonal_order,
                        dates=model_dict['history'].dt, 
                        freq=str(model_dict['fcst_interval']) + dt_units)
  
  sarimax_fit = sarimax_model.fit()
  model_summary = sarimax_fit.summary()
  model_residuals = pd.DataFrame({'residual': sarimax_fit.resid})
  model_residuals.reset_index(inplace=True)
  model_coefficients = pd.read_html(model_summary.tables[1].as_html(), 
                                    header=0, index_col=0)[0]

  model_coefficients['regressor'] = model_coefficients.index
  
  model_attributions = pd.DataFrame()
  model_attributions['dt'] = model_dict['history'].dt
  model_attributions['total'] = (
      model_dict['history'].actual - model_residuals.residual)
  model_attributions['actual'] = model_dict['history'].actual
  
  for x_col in model_dict['x_reg'].columns:
    x_col_mask = (model_coefficients.regressor == x_col)
    x_coef = float(model_coefficients.coef.loc[x_col_mask])
    x_var = float(model_coefficients.loc[x_col_mask, 'std err']) ** 2
    model_attributions[x_col] = x_coef * model_dict['x_reg'][x_col]
    model_attributions[x_col + '_var'] = (
        x_var * model_dict['x_reg'][x_col] ** 2)
    
  sarimax_fcst = sarimax_fit.get_forecast(
      model_dict['len_fcst'], exog=model_dict['x_future'])
                 
  sarimax_fcst_df = pd.DataFrame({'forecast':sarimax_fcst.predicted_mean})

  sarimax_pred_int_df = sarimax_fcst.conf_int(alpha=(1-pred_level)).rename(
      columns={'lower actual':'forecast_lower', 
               'upper actual':'forecast_upper'})

  sarimax_fcst_df = pd.concat([sarimax_fcst_df, sarimax_pred_int_df], axis=1)
  sarimax_fcst_df.reset_index(inplace=True)
  sarimax_fcst_df = sarimax_fcst_df.rename(columns={'index':'dt'})
  sarimax_fcst_bcktrans = _back_transform_df(sarimax_fcst_df, transform)
  
  return {'forecast':sarimax_fcst_bcktrans, 
          'transformed_forecast':sarimax_fcst_df,
          'coefficients':model_coefficients,
          'summary':model_summary,
          'residuals':model_residuals,
          'attributions':model_attributions}


def sarimax_111_011(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get SARIMAX model of order (1,1,1), and seasonal order (0,1,1)
     results.
   
   Parameters
   ----------
   history: pd dataframe containing the training history of the time series
       being modelled.
   dt_span: dict of the forecast begin and end dts
   dt_units: units of the original input time series
   periods: seasonal periods (handled using built in seasonality estimation
       methods intrinsic to each model in the ensemble - e.g. arima methods)
   periods_agg: seasonal periods that have been aggregated across
   periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
   pred_level: confidence level of the prediction intervals
   transform: transformation ('log' or 'none') applied to the timeseries being
      modelled
   x_reg: pandas dataframe of historical external regressors
   x_future: pandas dataframe of future values of external regressors
   holidays_df: pandas dataframe of holidays. Must have dt and holiday column
   x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
     with seasonality and holiday features
   
   Returns
   -------
  Dict containing dataframes for the model coefficients/summary and the 
     backtransformed forecast."""  
  return sarimax_pdq_PDQ(pdq_order=(1,1,1), s_pdq_order=(0,1,1),
                         history=history, 
                         dt_span=dt_span, dt_units=dt_units,
                         periods=periods, periods_agg=periods_agg, 
                         periods_trig=periods_trig, 
                         pred_level=pred_level, transform=transform,
                         x_reg=x_reg, x_future=x_future,
                         holidays_df=holidays_df,
                         x_cols_seasonal_interactions=
                         x_cols_seasonal_interactions)
 

def sarimax_013_011(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get SARIMAX model of order (0,1,3), and seasonal order (0,1,1)
     results.

  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
     backtransformed forecast."""  
  return sarimax_pdq_PDQ(pdq_order=(0,1,3), s_pdq_order=(0,1,1),
                         history=history, 
                         dt_span=dt_span, dt_units=dt_units,
                         periods=periods, periods_agg=periods_agg, 
                         periods_trig=periods_trig, 
                         pred_level=pred_level, transform=transform,
                         x_reg=x_reg, x_future=x_future,
                         holidays_df=holidays_df,
                         x_cols_seasonal_interactions=
                         x_cols_seasonal_interactions)                         

def sarimax_003_001(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get SARIMAX model of order (0,0,3), and seasonal order (0,0,1)
     results.

  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
     backtransformed forecast."""  
  return sarimax_pdq_PDQ(pdq_order=(0,0,3), s_pdq_order=(0,0,1),
                         history=history, 
                         dt_span=dt_span, dt_units=dt_units,
                         periods=periods, periods_agg=periods_agg, 
                         periods_trig=periods_trig, 
                         pred_level=pred_level, transform=transform,
                         x_reg=x_reg, x_future=x_future,
                         holidays_df=holidays_df,
                         x_cols_seasonal_interactions=
                         x_cols_seasonal_interactions)                         



def sarimax_002_001(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get SARIMAX model of order (0,0,2), and seasonal order (0,0,1)
     results.

  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
     backtransformed forecast."""  
  return sarimax_pdq_PDQ(pdq_order=(0,0,2), s_pdq_order=(0,0,1),
                         history=history, 
                         dt_span=dt_span, dt_units=dt_units,
                         periods=periods, periods_agg=periods_agg, 
                         periods_trig=periods_trig, 
                         pred_level=pred_level, transform=transform,
                         x_reg=x_reg, x_future=x_future,
                         holidays_df=holidays_df,
                         x_cols_seasonal_interactions=
                         x_cols_seasonal_interactions)                         

  
def sarimax_001_001(history,
                    dt_span,
                    dt_units='D',
                    periods=[1],
                    periods_agg=[7],
                    periods_trig=[365.25 / 7],
                    pred_level=0.8,
                    transform='none',
                    x_reg=None,
                    x_future=None,
                    holidays_df=None,
                    x_cols_seasonal_interactions=[]):
  """Function to get SARIMAX model of order (0,0,1), and seasonal order (0,0,1)
     results.

  Parameters
  ----------
  history: pd dataframe containing the training history of the time series
      being modelled.
  dt_span: dict of the forecast begin and end dts
  dt_units: units of the original input time series
  periods: seasonal periods (handled using built in seasonality estimation
      methods intrinsic to each model in the ensemble - e.g. arima methods)
  periods_agg: seasonal periods that have been aggregated across
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  pred_level: confidence level of the prediction intervals
  transform: transformation ('log' or 'none') applied to the timeseries being
     modelled
  x_reg: pandas dataframe of historical external regressors
  x_future: pandas dataframe of future values of external regressors
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
  x_cols_seasonal_interactions: list of columns in x_reg/x_future to interact
    with seasonality and holiday features
  
  Returns
  -------
  Dict containing dataframes for the model coefficients/summary and the 
     backtransformed forecast."""  
  return sarimax_pdq_PDQ(pdq_order=(0,0,1), s_pdq_order=(0,0,1),
                         history=history, 
                         dt_span=dt_span, dt_units=dt_units,
                         periods=periods, periods_agg=periods_agg, 
                         periods_trig=periods_trig, 
                         pred_level=pred_level, transform=transform,
                         x_reg=x_reg, x_future=x_future,
                         holidays_df=holidays_df,
                         x_cols_seasonal_interactions=
                         x_cols_seasonal_interactions)                         


def validate_parameters(dt_units='D',
                        transform='none',
                        periods=None,
                        periods_trig=[365],
                        holidays_df=None,
                        models=[sarimax_111_011,
                                sarimax_013_011,
                                prophet_linear_lt,
                                auto_sarimax,
                                auto_scarf_nnet],
                        consensus_method=np.median,
                        pred_level=0.8):
  """Function to set a dict of Aggregation parameters.
 
  Parameters
  ----------
  transform: transformation ('log' or 'none') applied to the timeseries being
    modelled
  periods: seasonal periods (handled using built in seasonality estimation
    methods intrinsic to each model in the ensemble - e.g. seasonal dummy vars)
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  models: list of models to use in the ensemble
  consensus_method: function to apply to model ensemble to get a consensus 
    forecast
  pred_level: confidence level of the prediction intervals for each model
  
  Returns
  -------
  Boolean indicating the ensemble parameters are valid.""" 
  if dt_units not in ['min', 'H', 'D', 'W', 'MS']:
    raise Exception('Invalid dt_units input. Must be "min", "H", "D", "W",' +
                    'or "MS"')
  
  if transform not in ['log', 'none']:
    raise Exception('transform options are "log" or "none".')

  if periods is not None and periods is not []:
    if len([period for period in periods if 
            period - np.round(period) == 0]) != len(periods):
      raise Exception("Seasonal periods must be positive-integer valued")
        
  if len(periods_trig) > 0:
    if (all(type(period_trig) not in [float, int, np.float64] 
            for period_trig in  periods_trig)):
      raise Exception("Trigonometric seasonal periods must be float, int or" +
                      "numpy.float64.")

  if (holidays_df is not None and 
      type(holidays_df) is not pd.core.frame.DataFrame):
    raise Exception("holidays_df must be a pandas dataframe.")
    
    if 'dt' not in holidays_df.columns | 'holiday' not in holidays_df.columns:
      raise Exception("holidays_df must have columns 'dt' and 'holiday'")
    
    if search('timestamp', str(type(holidays_df['dt'][0])).lower()) is None:
      raise Exception("holidays_df column dt must be a timestamp.")
  
  # Check that each model specified has the required arguments
  required_model_args = ['history', 'dt_span', 'dt_units',
                         'periods', 'periods_agg', 'periods_trig',
                         'pred_level', 'transform',
                         'x_reg', 'x_future', 'holidays_df',
                         'x_cols_seasonal_interactions']
  for model in models:
    if any([model_arg != required_model_arg 
           for model_arg, required_model_arg in 
           zip(getfullargspec(model).args, required_model_args)]):
      raise Exception("Model" + str(model) + 'does not have required ' + 
                      'arguments: "' + '", "'.join(required_model_args) + '"')
      
  if consensus_method is not np.median and consensus_method is not np.mean:
    raise Exception("Current consensus method options are np.median or " +
                    "np.mean")
  
  if pred_level < 0 or pred_level > 1: 
    raise Exception('pred_level must be between 0 and 1')

  return True
  
