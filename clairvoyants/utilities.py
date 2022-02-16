# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from clairvoyants.aggregation import aggregate_to_longest

from dateutil.relativedelta import relativedelta

import math
import numpy as np
import pandas as pd

def _datetime_delta(delta,
                    dt_units='D'):
  """Internal function to more neatly generate a time delta given a time unit
  (and one that includes a time unit of months)
   
  Parameters
  ----------
  delta: integer giving the number of dt_units to add
  dt_units: the datetime units for the datetime delta, options are 'min', 'H',
  'D', 'W', 'MS'
  
  Returns
  -------
  A dict containing a new begin_dt and end_dt."""
  if dt_units not in ['min', 'H', 'D', 'W', 'MS', 'Y']:
    raise Exception('Invalid dt_units input. Must be "min", "H", "D", "W",' +
                    '"MS", or "Y"')
  
  dt_delta = relativedelta(minutes=int(dt_units == 'min') * delta,
                           hours=int(dt_units == 'H') * delta,
                           days=int(dt_units == 'D') * delta,
                           weeks=int(dt_units == 'W') * delta,
                           months=int(dt_units == 'MS') * delta,
                           years=int(dt_units == 'Y') * delta)
  return dt_delta


def _get_begin_end_dts(begin_dt,
                       end_dt,
                       dt_units='D',
                       fix_dt_option='begin_later',
                       need_integer_period=True,
                       period=7):
  """Function to reset beginning or end date of training history or forecast 
  period to give an integer number of periods (if aggregating over a period).
  Note: if you feed begin_dt and end_dt that do not align with dt_units it will
  not return expected results. 
   
  Parameters
  ----------
  begin_dt: candidate start datetime
  end_dt: candidate end datetime (inclusive)
  dt_units: time units of the input series (unaggregated)
  fix_dt_option whether to cut data to the right or left (end_later vs
    begin_later)
  need_integer_period: whether an integer number of periods are needed (if
    aggregating)
  period: length of maximal period being aggregated over
   
  Returns
  -------
  A dict containing a new begin_dt and end_dt."""
 
  if begin_dt is None or end_dt is None:
    raise Exception("Initial beginning and end date must be supplied")

  if not need_integer_period:
    return {'begin_dt':begin_dt, 'end_dt':end_dt}

  l_d = len(pd.to_datetime(pd.date_range(begin_dt, end_dt, freq=dt_units)))
  seq_d = list(range(1, (period + 1))) * int(math.ceil(l_d / period))
  begin_d = seq_d[0]
  end_d = seq_d[(l_d - 1)] + 1
  
  if begin_d is not end_d:
    delta = end_d - begin_d

    if delta < 0:
      delta += period
    
    if fix_dt_option == 'begin_later':
      begin_dt += _datetime_delta(delta, dt_units)
      
    else:
      end_dt += _datetime_delta(period - delta, dt_units)
  
  if (len(pd.to_datetime(pd.date_range(begin_dt, end_dt, freq=dt_units))) %
      period != 0):
    raise Exception("Error occurred. No integer number of periods." + 
                    "Check your input.")
    
  return {'begin_dt':pd.to_datetime(begin_dt),
          'end_dt':pd.to_datetime(end_dt)}
  

def _transform_timeseries(timeseries, 
                          transform='none'):
  """Internal function to return a transformed time series
   
  Parameters
  ----------
  timeseries: pd series to be transformed
  transform: transformation to apply to the series. Options are 'none' and 
    'log' (and eventually 'box_cox')
  
  Returns
  -------
  A dict containing a new begin_dt and end_dt."""
  if transform not in ['log', 'none']:
    raise Exception('transform options are "log" or "none".')
  
  if transform == 'log':
    timeseries = np.log(timeseries)
    
  return timeseries
    

def _back_transform_timeseries(timeseries,
                               transform='none'):
  """Internal function to return a back transformed time series
   
  Parameters
  ----------
  timeseries: pd series to be back transformed
  transform: transformation to apply to the series. Options are 'none' and 
    'log' (and eventually 'box_cox')
    
  Returns
  -------
  A backtransformed pd series"""
  if transform not in ['log', 'none']:
    raise Exception('transform options are "log" or "none".')
  
  if transform == 'log':
    timeseries = np.exp(timeseries)
    
  return timeseries
    

def _aggregate_and_transform(history,
                             periods_agg=[7],
                             agg_fun=['sum'],
                             cols_agg=['actual'],
                             transform='none'):

  """Function to aggregate over seasonal periods and transform the aggregated
   series in preparation to be modelled.
  
  Parameters
  ----------
  history: pandas dataframe containing dt and actual
  periods_agg: list of periods to aggregate over
  agg_fun: function(s) to apply to columns being aggregated
  cols_agg: columns of history to aggregate
  transform: transformation to apply to the series. Options are 'none' and 
    'log' (and eventually 'box_cox')
  
  Returns
  -------
  Named dict containing aggregated histories as pandas dataframes for each
  period aggregated over and the transformed timeseries for the longest
  aggregated seasonal period""" 
  if len(periods_agg) > 0 and max(periods_agg) > 1:
    aggregated_histories = aggregate_to_longest(history, periods_agg, agg_fun,
                                                cols_agg)
    period_str = 'period' + str(max(periods_agg))
    transformed_history = aggregated_histories[period_str].copy()
  else:
    aggregated_histories = None
    transformed_history = history.copy()


  for col_trans in cols_agg:
    transformed_history[col_trans] = _transform_timeseries(
      transformed_history[col_trans], transform)

  return {'aggregated':aggregated_histories,
          'transformed':transformed_history}


def _back_transform_df(df,
                       transform='none',
                       cols_transform=['forecast', 
                                       'forecast_lower', 
                                       'forecast_upper']):
  """Internal function to return a back transformed time series
  
  Parameters
  ----------
  df: pd df to be back transformed
  transform: transformation to apply to the series. Options are 'none' and 
    'log' (and eventually 'box_cox')
  cols_transform: list of the columns to back transform
  
  Returns
  -------
  A backtransformed pd df"""
  df_backtransformed = df.copy()
  
  for col_transform in cols_transform:
    df_backtransformed[col_transform] = _back_transform_timeseries(
        df[col_transform], transform)

  return df_backtransformed


def _scale_history(history):
  """Internal function to return a scaled history. 
  
  Parameters
  ----------
  history: 
  
  Returns
  -------
  The history pd dataframe with standardized actuals column appended"""
 
  history['actual_scaled'] = ((history['actual'] - history['actual'].mean())
                              / history['actual'].std())
  return history

def _rescale_forecast(forecast, 
                      history, 
                      cols_rescale=['forecast', 
                                    'forecast_lower', 'forecast_upper']):
  """Internal function to return a rescaled forecast resulting from a scaled
     history. 

  Parameters
  ----------
  history: the history that generated the forecast
      
  Returns
  -------
  The forecast pd dataframe with the forecast columns rescaled"""
  hist_mean = history['actual'].mean()
  hist_std = history['actual'].std()
  for col_rescale in cols_rescale:
    forecast[col_rescale] = forecast[col_rescale] * hist_std + hist_mean

  return forecast

    
def _scale_features(x_reg, 
                    x_future,
                    cols_scale=None):
  """Internal function to return a scaled feature set. 
  
  Parameters
  ----------
  x_reg: pd dataframe containing historical features to standardize
  x_future: pd dataframe containing future features to standardize
  cols_scale: list of the columns to standardize
  
  Returns
  -------
  A dict of pd dataframes with standardized historical and future features"""
  
  if cols_scale is None: 
    cols_scale = x_reg.columns
   
  x_all = pd.concat([x_reg, x_future])
  
  for col_scale in cols_scale:
    if len(np.unique(x_all[col_scale])) > 10:
      x_all[col_scale] = ((x_all[col_scale] - x_all[col_scale].mean()) 
                          / x_all[col_scale].std())
      
    elif any([type(val) in [np.float_, float] 
                 for val in np.unique(x_all[col_scale])]):
      x_all[col_scale] = (x_all[col_scale] - x_all[col_scale].mean())
        
  x_reg_scaled = x_all.iloc[:x_reg.shape[0]]
  x_future_scaled = x_all.iloc[x_reg.shape[0]:]
  
  return {'x_reg':x_reg_scaled, 'x_future':x_future_scaled}


def _diff_history(history, d):
  """Internal function to return the history diffed at order d
 
  Parameters
  ----------
  history: pd dataframe with actual to difference
  d: order of differencing to apply
   
  Returns
  -------
  history appended with a differenced actual column, actual_diff"""
  
  history['actual_diff'] = history.actual
  
  for i in list(range(d)):
      
    history.actual_diff = history.actual_diff - history.shift(1).actual_diff
    
  return history
 
def _int_forecast(forecast, history, d, cols_int=['forecast',
                                                  'forecast_lower',
                                                  'forecast_upper']):
  """Internal function to return the history diffed at order d
 
  Parameters
  ----------
  forecast: pd dataframe with forecast columns to integrate
  history: pd dataframe with actuals that were differenced
  d: order of differencing that was applied
  
   
  Returns
  -------
  Forecast integrated"""
  history['actual_int'] = history['actual_diff']
  for i in list(range(d)):

    history['actual_int'] += history.shift(1).actual_int
    print(history)
    
    for col_int in cols_int:  
    
      forecast[col_int] += (list(
          history.shift(i).actual_int[history.shape[0] - 1:]) + list(
           forecast.shift(1)[col_int])) 
    
  return forecast
    
   

def _percentile(q):
  """Internal function to return percentile for use in pandas.groupby().agg()
   
 Parameters
  ----------
  q: quantile to return
   
 Returns
  -------
  A percentile function compatible with pandas.groupby().agg()"""

  def percentile_(x):
    return np.percentile(x, q)
  percentile_.__name__ = 'percentile_%s' % q
  return percentile_
