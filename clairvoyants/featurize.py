# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from clairvoyants.aggregation import aggregate_to_longest
from clairvoyants.utilities import _datetime_delta, _scale_features, _scale_history, _diff_history

import numpy as np
import pandas as pd
from pmdarima import auto_arima


def featurize_holidays(history,
                       holidays_df):
  """Function to obtain holiday features as dummy variables to use as external 
  regressors.
   
  Parameters
  ----------
  history: pd dataframe to get holiday features for
  holidays_df: pd dataframe containing dt and holiday columns to featurize
   
  Returns
  -------
  pd dataframe of dummy variables for each holiday for the dts in history."""
  
  holidays_featurized = history[['dt']]
  unique_holidays = list(set(holidays_df['holiday']))
  
  for hday in unique_holidays:
   
    holiday_dts = holidays_df[holidays_df['holiday'] == hday] 
    holidays_featurized[hday] = [int(dt in holiday_dts['dt'].to_list())
                                 for dt in holidays_featurized['dt']]
    
  return holidays_featurized
  

def get_holiday_features(history_start,
                         fcst_dts,
                         holidays_df,
                         dt_units='D',
                         periods_agg=[7]):
  """Function to get holiday features as dummy variables to use as external 
  regressors and predictors (and aggregate if necessary).
   
  Parameters
  ----------
  history_start: start datetime (dt) of history to build holiday features for
  fcst_dts: dict containing the forecast start and end dt 
  dt_units: the datetime units of the input timeseries: H, D, W, M, Y
  periods_agg: periods that are aggregated over
  holidays_df: pd df containing a column for dt and holiday
   
  Returns
  -------
  Dict of pd dataframes containing featurized holidays for the history and
  forecast periods."""
  
  total_dts = pd.DataFrame()
  total_dts['dt'] = pd.to_datetime(pd.date_range(
     start=history_start, end=fcst_dts['end_dt'],
     freq=dt_units))

  holidays_featurized = featurize_holidays(total_dts, holidays_df)

  holidays_all = list(set(holidays_df['holiday']))
  
  if (len(periods_agg) > 0 and max(periods_agg) > 1):
      
    holidays_featurized = aggregate_to_longest(history=holidays_featurized,
                                               periods_agg=periods_agg,
                                               agg_fun=['max'],
                                               cols_agg=holidays_all)
    
    m_period_str = 'period' + str(max(periods_agg))
    holidays_featurized = holidays_featurized[m_period_str]

  holidays_featurized = holidays_featurized.set_index('dt')
  holidays_past = holidays_featurized[holidays_featurized.index < 
                                      fcst_dts['begin_dt']]
  holidays_future = holidays_featurized[holidays_featurized.index >=
                                        fcst_dts['begin_dt']]
  
  holidays_featurized_dict = {}
  holidays_featurized_dict['past'] = holidays_past
  holidays_featurized_dict['future'] = holidays_future
  
  return holidays_featurized_dict


def get_trig_seasonality_features(history_len, 
                                  fcst_len,
                                  periods_trig=[365]):

  """Function to obtain trigonometric seasonality features to use as external 
  regressors and predictors (in place of built in dummy variable or other
  seasonality).
  
  Parameters
  ----------
  history_len: length of series (or number of rows of history df)
  fcst_len: length of the desired forecast
  periods_trig: list of seasonal periods to create trigonometric features for
  
  Returns
  -------
  Dict of pd dataframes containing featurized trig seasonality for the history
  and forecast periods."""
  
  trig_seasonality_features = pd.DataFrame()
  trig_seasonality_features['idx'] = range(1, history_len + fcst_len + 1)
  for period in periods_trig:

    period_index = trig_seasonality_features['idx'] % period

    period_sin_str = 'sin_period' + str(period)
    trig_seasonality_features[period_sin_str] = np.sin(2 * np.pi / period *
                                                       period_index)
    
    period_cos_str = 'cos_period' + str(period)
    trig_seasonality_features[period_cos_str] = np.cos(2 * np.pi / period *
                                                       period_index)
    
  trig_seasonality_past = trig_seasonality_features[
     trig_seasonality_features['idx'] <= history_len].drop(['idx'], axis=1)
    
  trig_seasonality_future = trig_seasonality_features[
      trig_seasonality_features['idx'] > history_len].drop(['idx'], axis=1)
  
  trig_seasonality_dict = {}
  
  trig_seasonality_dict['past'] = trig_seasonality_past
  
  trig_seasonality_dict['future'] = trig_seasonality_future
  
  return trig_seasonality_dict


def _get_ar_diff_order(history, period_ts, len_fcst, dt_span_seq,
                       scale_history=False, 
                       diff_history=False, auto_create_lags=True,
                       x_reg=None, x_future=None,
                       max_p=3, max_P=1, max_d=3):
  """Function automatically find the order of historical lags to include as 
     features and the order of differencing to apply
  
  Parameters
  ----------
  history: pd dataframe containing ts to model (actual)
  period_ts: seasonal period of the timeseries
  len_fcst: the length of the forecast
  scale_history: whether to standardize the actuals 
  diff_history: whether the history will be diffed based on the auto_arima
    optimal d (thus whether to even find optimal d)
  auto_create_lags: whether lags of the history will be created as features
  x_reg: pd dataframe of historical external regressors
  x_future: pd dataframe of future values of external regressors
  max_p: the maximal order of the nonseasonal lags to be created
  max_P: the maximal order of the seasonal lags to be created
  max_d: the maximal order of differencing to be applied  
  
  Returns
  -------
  Dict containing the auto_arima determined pdq order and seasonal PDQ
    order as well as the initial forecast."""

  col_ts = 'actual' + scale_history * '_scaled'
  
  auto_sarimax_model = auto_arima(history[col_ts], X=x_reg.to_numpy() if x_reg is not None else None,
      max_P=max_P * auto_create_lags, max_p=max_p * auto_create_lags,
      stationary=(not diff_history), max_d=max_d * diff_history, max_D=1,
      with_intercept=True, max_q=0, start_q=0, max_Q=0, start_Q=0,
      method='bfgs', stepwise=False, m=period_ts, 
      start_p=int(auto_create_lags), start_P=int(auto_create_lags), d=int(diff_history), D=0,
      max_order=max_p + max_d)
  
  auto_sarimax_fcst = auto_sarimax_model.predict(
      len_fcst, X=x_future.to_numpy() if x_future is not None else None, 
      return_conf_int=True)
  
  auto_sarimax_fcst_df = pd.DataFrame(
      {'dt':dt_span_seq,
       'forecast':auto_sarimax_fcst[0][:len_fcst]})
  
  return {'order':auto_sarimax_model.order,
          'seasonal_order':auto_sarimax_model.seasonal_order,
          'forecast':auto_sarimax_fcst_df}
  

def featurize_lags(history, forecast, period_ts,
                   scale_history=False, diff_history=False, 
                   x_reg=None, x_future=None, p_ar=1, P_ar=0):
  """Function create lags of the history as features
  
  Parameters
  ----------
  history: pd dataframe containing ts to model (actual)
  forecast: an initial forecast to build lags for x_future
  scale_history: whether to standardize the actuals
  diff_history: whether to diff the history based on the auto_arima estimated
    optimal d
  period_ts: seasonal period of the timeseries
  x_reg: pd dataframe of historical external regressors
  x_future: pd dataframe of future values of external regressors
  p_ar: the order of the nonseasonal lags to create
  P_ar: the order of the seasonal lags to create  
  
  Returns
  -------
  Dict of pd dataframes containing historical and future values of the
     external regressors."""

  if scale_history and diff_history:
    raise Exception('The history cannot be scaled and differenced.')

  col_ts = 'actual' + scale_history * '_scaled' + diff_history * '_diff'
  if x_reg is None or x_future is None:
    x_reg = pd.DataFrame()
    x_future = pd.DataFrame()
  if p_ar > 0:
    for p in list(range(1, p_ar + 1)):
      x_reg['actual_lag' + str(p)] = history.shift(p)[col_ts]
      x_future['actual_lag' + str(p)] = list(
              history[col_ts][history.shape[0] - p:]) + list(
                  forecast.shift(p).forecast[p:])
    
    if P_ar > 0:
    
      x_reg['actual_seasonal_lag' + str(period_ts)] = history.shift(
         period_ts)[col_ts]
      x_future['actual_seasonal_lag' + str(period_ts)] = list(
            history.actual[history.shape[0] - period_ts:]) + list(
                forecast.shift(period_ts).forecast[period_ts:])
  
  return {'x_reg':x_reg, 'x_future':x_future}


def _process_features(history, scale_history=False, diff_history=False,
                      dt_span=None, dt_units='D', 
                      periods=[], periods_agg=[7], periods_trig=[365.25 / 7],
                      holidays_df=None, 
                      x_reg=None, x_future=None, scale_x=False,
                      auto_create_lags=False,
                      x_cols_seasonal_interactions=[]):
  """Internal function to process history and x features (e.g. add 
     trigonometric seasonality and holiday features or 
     standardize continuous columns)
  
  Parameters
  ----------
  history: pd dataframe containing ts to model (actual)
  scale_history: whether to standardize the actuals
  diff_history: whether to diff the history based on the auto_arima estimated
    optimal d
  dt_span: dict of the forecast begin_dt and end_dt
  dt_units: the units of the input ts (before aggregation)
  periods: seasonal periods of the timeseries
  periods_agg: seasonal periods that were aggregated over
  periods_trig: seasonal periods to get trigonometric features for
  holidays_df: pd dataframe of holidays to featurize 
  x_reg: pd dataframe of historical external regressors
  x_future: pd dataframe of future values of external regressors
  scale_x: whether to standardize the input external regressors dataframes
  auto_create_lags: whether to find the optimal int of seasonal and nonseasonal
    lags and append these to the dataframe
  x_cols_seasonal_interactions: columns of the x_features dfs to interact with
   any seasonal or holiday effects
  
  Returns
  -------
  Dict of pd dataframes containing processed history, external regressors and
    other relevant metrics."""
  
  fcst_interval = int(max(periods_agg + [1]))
  dt_span_seq = pd.to_datetime(pd.bdate_range(
      str(dt_span['begin_dt'] + _datetime_delta(fcst_interval - 1, dt_units)),
      str(dt_span['end_dt']),
      freq=str(fcst_interval) + dt_units))
  history_l = history.copy()
  len_fcst = len(dt_span_seq)
  history_l.index = history_l['dt']
  
   
  if len(periods) > 0:
    periods = [p for p in periods if p > 1]

  if len(periods) > 0:
    period_ts = int(max(periods))

  else:
    period_ts = 1
    
  
  if ((x_reg is not None and x_future is not None) or holidays_df is not None 
      or len(periods_trig) > 0):
      
    if x_reg is not None and x_future is not None:
    
      x_reg_l = x_reg.copy()
      x_future_l = x_future.copy()
      
      if scale_x:
        
        x_all = _scale_features(x_reg_l, x_future_l, x_reg_l.columns)
        x_reg_l = x_all['x_reg']
        x_future_l = x_all['x_future']
    
    if holidays_df is not None:
        
      history_begin = min(history['dt']) - _datetime_delta(fcst_interval - 1,
                                                           dt_units)
      holiday_features = get_holiday_features(history_begin,
                                              dt_span, holidays_df,
                                              dt_units, periods_agg) 
      
      if len(x_cols_seasonal_interactions) > 0:
        for x_col in x_cols_seasonal_interactions:
          holiday_features_past_inter = holiday_features['past'].multiply(
              x_reg_l[x_col], axis=0).add_suffix('_x_' + x_col)
          holiday_features['past'] = pd.concat(
              [holiday_features['past'].reset_index(drop=True),
               holiday_features_past_inter.reset_index(drop=True)], axis=1)
          holiday_features_fut_inter = holiday_features['future'].multiply(
              x_future_l[x_col], axis=0).add_suffix('_x_' + x_col)
          holiday_features['future'] = pd.concat(
              [holiday_features['future'].reset_index(drop=True),
               holiday_features_fut_inter.reset_index(drop=True)], axis=1)
      
      if (x_reg is not None and x_future is not None):
            
        holiday_features['past'].index = x_reg_l.index
        x_reg_l = pd.concat([x_reg_l, holiday_features['past']], axis=1)
        holiday_features['future'].index = x_future_l.index
        x_future_l = pd.concat([x_future_l, holiday_features['future']],
                               axis=1)
      else:
          
        x_reg_l = holiday_features['past']
        x_future_l = holiday_features['future']
        x_reg_l.index = history_l.index
        x_future_l.index = dt_span_seq
    
    if len(periods_trig) > 0 and max(periods_trig) > 1:
        
      trig_seasonality_features = get_trig_seasonality_features(
          history.shape[0], len_fcst, periods_trig)
      
      if len(x_cols_seasonal_interactions) > 0:
        for x_col in x_cols_seasonal_interactions:
            
          trig_features_past_int = (
              trig_seasonality_features['past'].reset_index(
                  drop=True).multiply(x_reg_l.reset_index(drop=True)[x_col], 
                                      axis=0).add_suffix('_x_' + x_col))

          trig_seasonality_features['past'] = pd.concat(
              [trig_seasonality_features['past'].reset_index(drop=True),
               trig_features_past_int.reset_index(drop=True)], axis=1)

          trig_features_fut_int = (
              trig_seasonality_features['future'].reset_index(
                  drop=True).multiply(x_future_l.reset_index(drop=True)[x_col],
                                      axis=0).add_suffix('_x_' + x_col))
          trig_seasonality_features['future'] = pd.concat(
              [trig_seasonality_features['future'].reset_index(drop=True),
               trig_features_fut_int.reset_index(drop=True)], axis=1)

    
      if x_reg is not None and x_future is not None:

        
        trig_seasonality_features['past'].index = x_reg_l.index
        x_reg_l = pd.concat([x_reg_l, trig_seasonality_features['past']],
                            axis=1)
        trig_seasonality_features['future'].index = x_future_l.index
        x_future_l = pd.concat([x_future_l, 
                                trig_seasonality_features['future']], axis=1)
        
      else:
          
        x_reg_l = trig_seasonality_features['past']
        x_future_l = trig_seasonality_features['future']
        x_reg_l.index = history_l.index
        x_future_l.index = dt_span_seq
        
  
  if scale_history:
    history_l = _scale_history(history_l)
  
  if diff_history or auto_create_lags:
      
    ar_diff = _get_ar_diff_order(history_l, period_ts, len_fcst, dt_span_seq,
                                 scale_history, diff_history, auto_create_lags,
                                 x_reg_l, x_future_l)
    pdq_order = ar_diff['order']
    seasonal_order = ar_diff['seasonal_order']
    
    if diff_history:
      history_l = _diff_history(history_l, pdq_order[2])
    
    if auto_create_lags:
        
      x_all = featurize_lags(history_l, ar_diff['forecast'], period_ts,
                             scale_history, diff_history, 
                             x_reg_l, x_future_l,
                             pdq_order[0], seasonal_order[0])
      
      x_reg_l = x_all['x_reg']
      x_future_l = x_all['x_future']
  
    trunc_idx = int(max([pdq_order[0] + pdq_order[2],
                         seasonal_order[0] * period_ts + pdq_order[2]]))

    x_reg_l = x_reg_l.iloc[trunc_idx:]
    history_l = history_l.iloc[trunc_idx:]
    
  else:
    pdq_order=None
    seasonal_order=None
    

  return {'history': history_l, 'fcst_interval':fcst_interval,
          'len_fcst':len_fcst, 
          'dt_span_seq':dt_span_seq, 'period_ts':period_ts,
          'x_reg':x_reg_l, 'x_future':x_future_l,
          'pdq_order':pdq_order, 'seasonal_order':seasonal_order}


