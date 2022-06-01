# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from clairvoyants.aggregation import aggregate_to_longest
from clairvoyants.utilities import _datetime_delta

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def prepare_history_to_disaggregate(history,
                                    aggregated_history,
                                    period_agg=7,
                                    x_reg=None,
                                    x_cols_seasonal_interactions=[]):
  """Function to prepare unaggregated histories to be used as training data
     in a model to used to disaggregate a forecast.
  
   Parameters
  ----------
  history: pandas dataframe containing unaggregated dt and actual
  aggregated_history: pandas dataframe containing the aggregated history
  period_agg: period that aggregated_history is aggregated over
  x_reg: optional matrix of x_features containing columns to include in 
    modelling disaggregation proportions
  x_cols_seasonal_interactions: columns of x_features to interact with 
    seasonality

  
  Returns
  -------
  pd data frame containing the unaggregated history as proportions of the  
  aggregated history for each period of length period_agg. It also includes
  the logit of this proportion and any x_features interacted with dummy 
  variables for the aggregated period in the unaggregated data"""
  
  if history.shape[0] % period_agg != 0:
    raise Exception('history must be an integer number of period_agg')

  hist = history.copy()
  hist['aggregated_actual'] = np.repeat(list(aggregated_history.actual),
                                        period_agg)
  hist['proportion_aggregated'] = hist.actual / hist.aggregated_actual
  hist['p'] = list(range(1, period_agg + 1)) * int(hist.shape[0] / period_agg)
   
  hist['logit_proportion_aggregated'] = np.log(
      hist['proportion_aggregated'] / (1 - hist['proportion_aggregated']))
  
  dummy_p = pd.get_dummies(hist['p'], prefix='p')
  
  if x_reg is not None and len(x_cols_seasonal_interactions) > 0:
    x_reg = x_reg.reset_index(drop=True)
    for x_col in x_cols_seasonal_interactions:
    
      dummy_interact = pd.get_dummies(
          hist['p'].reset_index(drop=True), prefix=x_col + '_x_p').multiply(
                x_reg[x_col].reset_index(drop=True), axis=0)
  
      dummy_p = pd.concat([dummy_p.reset_index(drop=True),
                           dummy_interact.reset_index(drop=True)], axis=1)
  
  elif x_reg is None:
    x_reg = hist['p']
        
  hist = pd.concat([hist.reset_index(drop=True),
                    x_reg.reset_index(drop=True),
                    dummy_p.reset_index(drop=True)], axis=1)
  hist = hist.drop(columns=['p'])

  return hist

def prepare_forecast_to_disaggregate(forecast_dates_disaggregated,
                                     aggregated_forecast,
                                     period_agg=7,
                                     x_future=None,
                                     x_cols_seasonal_interactions=[]):    
  """Function to prepare forecast data features to be used to disaggregate a 
     forecast.
  
   Parameters
  ----------
  aggregated_forecast: pandas dataframe containing the aggregated forecast
  period_agg: period that aggregated_forecast is aggregated over
  x_future: optional matrix of x_features containing columns to include in 
    modelling disaggregation proportions
  x_cols_seasonal_interactions: columns of x_future to interact with 
    seasonality
  
  Returns
  -------
  pd data frame containing the unaggregated forecast dates,   
  aggregated forecast and pi for each period of length period_agg. It also 
  includes any x_features interacted with dummy 
  variables for the aggregated period in the unaggregated data"""
  
  if len(forecast_dates_disaggregated) % period_agg != 0:
    raise Exception('forecast dates must be an integer number of period_agg')

  fcst = pd.DataFrame()
  fcst['dt'] = forecast_dates_disaggregated
  fcst['aggregated_forecast'] = np.repeat(list(aggregated_forecast.forecast),
                                               period_agg)
  
  fcst['aggregated_forecast_lower'] = np.repeat(list(
      aggregated_forecast.forecast_lower), period_agg)
  fcst['aggregated_forecast_upper'] = np.repeat(list(
      aggregated_forecast.forecast_upper), period_agg)
  
  fcst['p'] = list(range(1, period_agg + 1)) * int(fcst.shape[0] / period_agg)

  dummy_p = pd.get_dummies(fcst['p'], prefix='p')
  
  if x_future is not None and len(x_cols_seasonal_interactions) > 0:
    x_future = x_future.reset_index(drop=True)
    for x_col in x_cols_seasonal_interactions:
    
      dummy_interact = pd.get_dummies(
          fcst['p'].reset_index(drop=True), prefix=x_col + '_x_p').multiply(
                x_future[x_col].reset_index(drop=True), axis=0)
      dummy_p = pd.concat([dummy_p.reset_index(drop=True),
                           dummy_interact.reset_index(drop=True)], axis=1)
      
  elif x_future is None:
    x_future = fcst['p']
        
  fcst = pd.concat([fcst.reset_index(drop=True),
                    x_future.reset_index(drop=True), 
                    dummy_p.reset_index(drop=True)], axis=1)

  fcst = fcst.drop(columns=['p'])
  
  return fcst


def disaggregate_forecast(history,
                          aggregated_history,
                          aggregated_forecast,
                          dt_units='D',
                          period_agg=7,
                          period_disagg=1,
                          x_reg=None,
                          x_future=None,
                          x_cols_seasonal_interactions=[],
                          pred_level=0.8,
                          agg_fun=['sum']):
  """Function to disaggregate an aggregated forecast using logit transformed
    ARIMA model.
 
  Parameters
  ----------
  history: pandas dataframe containing unaggregated dt and actual
  aggregated_history: pandas dataframe containing the aggregated history
  period_agg: period that aggregated_history is aggregated over
  x_features: optional matrix of x_features containing columns to include in 
    modelling disaggregation proportions
  x_cols_seasonal_interactions: columns of x_features to interact with 
    seasonality

 
  Returns
  -------
  pd data frame containing the disaggregated forecast and prediction 
  intervals"""
  if x_reg is not None:
    x_reg_p = x_reg.reset_index()
  else: 
    x_reg_p = x_reg
  hist = prepare_history_to_disaggregate(history, aggregated_history,
                                         period_agg, 
                                         x_reg_p,
                                         x_cols_seasonal_interactions)
  forecast_dates_disaggregated =  pd.to_datetime(pd.bdate_range(
      str(max(history.dt) + _datetime_delta(period_disagg, dt_units)),
      max(aggregated_forecast.dt),
      freq=str(period_disagg) + dt_units))
  
  if x_future is not None:
    x_future_p = x_future.reset_index()
  else: 
    x_future_p = x_future
  fcst = prepare_forecast_to_disaggregate(forecast_dates_disaggregated,
                                          aggregated_forecast,
                                          period_agg,
                                          x_future_p,
                                          x_cols_seasonal_interactions)
  
  per_dummies = ['p_' + str(p) for p in range(1, period_agg)]
  per_interactions = [x_col + '_x_p_' + str(p) 
                      for x_col in x_cols_seasonal_interactions
                      for p in list(range(1, period_agg))]
  
  if x_reg is not None and x_future is not None:
    x_cols = list(x_reg.columns)
  else:
    x_cols = []

  arima002_disagg_model = ARIMA(
      hist.set_index('dt').logit_proportion_aggregated, order=(0, 0, 2),
      exog=hist.set_index('dt')[x_cols + per_dummies + per_interactions],
      freq=str(period_disagg) + dt_units)
  
  arima002_disagg_fit = arima002_disagg_model.fit()
  disagg_model_summary = arima002_disagg_fit.summary()
  
  disagg_model_residuals = pd.DataFrame({ 
      'residual': arima002_disagg_fit.resid})
  
  disagg_model_coefficients = pd.read_html(
      disagg_model_summary.tables[1].as_html(), header=0, index_col=0)[0]

  disagg_model_coefficients['regressor'] = disagg_model_coefficients.index
  
  disagg_fcst = arima002_disagg_fit.get_forecast(
      len(forecast_dates_disaggregated), 
      exog=fcst.set_index('dt')[x_cols + per_dummies + per_interactions])
  
  disagg_fcst_df = pd.DataFrame({'forecast':disagg_fcst.predicted_mean})

  disagg_pred_int_df = disagg_fcst.conf_int(alpha=(1-pred_level)).rename(
      columns={'lower logit_proportion_aggregated':'forecast_lower', 
               'upper logit_proportion_aggregated':'forecast_upper'})

  disagg_fcst_df = pd.concat([disagg_fcst_df, disagg_pred_int_df], axis=1)
  disagg_fcst_df.reset_index(inplace=True)
  disagg_fcst_df = disagg_fcst_df.rename(columns={'index':'dt'})
  
  
  fcst_cols = ['forecast', 'forecast_lower', 'forecast_upper']
  for col in fcst_cols:
    disagg_fcst_df[col] = np.exp(disagg_fcst_df[col]) / (
        1 + np.exp(disagg_fcst_df[col]))


  agg_fcst_df = aggregate_to_longest(disagg_fcst_df, [period_agg],
                                     agg_fun, cols_agg=fcst_cols)
  agg_fcst_df = agg_fcst_df['period' + str(period_agg)]

  for col in fcst_cols:
    disagg_fcst_df[col] = disagg_fcst_df[col] / np.repeat(list(
        agg_fcst_df[col]), period_agg)
    disagg_fcst_df[col] = disagg_fcst_df[col] * fcst['aggregated_' + col]


  return {'summary':disagg_model_summary,
          'residuals':disagg_model_residuals,
          'coefficients':disagg_model_coefficients,
          'disaggregated':disagg_fcst_df}     

