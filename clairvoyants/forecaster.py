# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from clairvoyants import aggregation
from clairvoyants.disaggregation import disaggregate_forecast
from clairvoyants import ensemble
from clairvoyants.utilities import _aggregate_and_transform, _datetime_delta, _get_begin_end_dts, _back_transform_df

import numpy as np
import pandas as pd
from re import search

class Clairvoyant(object):
  """Clairvoyant ensemble forecaster.
  
  Parameters
  ----------
  
  dt_units: datetime units of the timeseries to be modelled
  periods_agg: list of periods to aggregate over
  agg_fun: function to apply to actuals when aggregating periods
  agg_fun_x_features: list of functions to apply to x_features when 
      aggregating periods
  transform: transformation ('log' or 'none') applied to the timeseries being
      modelled
  periods: seasonal periods (handled using built in seasonality estimation
    methods intrinsic to each model in the ensemble - e.g. seasonal dummy vars)
  periods_trig: seasonal periods to fit trigonometric curves (sin + cos) to
  models: list of models to use in the ensemble
  x_features: pandas dataframe of external regressors. Must have dt column
  holidays_df: pandas dataframe of holidays. Must have dt and holiday column
    """

  def __init__(self,
               dt_units='D',
               periods_agg=[7],
               agg_fun=['sum'],
               agg_fun_x_features=['sum'],
               transform='none',
               periods=[],
               periods_trig=[365.25],
               holidays_df=None,
               models=[ensemble.sarimax_111_011,
                       ensemble.sarimax_013_011,
                       ensemble.auto_sarimax,
                       ensemble.prophet_linear_lt,
                       ensemble.auto_sarif_nnet,
                       ensemble.auto_scarf_nnet],
               consensus_method=np.median,
               pred_level=0.8):
        
    
      if aggregation.validate_parameters(periods_agg, agg_fun,
                                         agg_fun_x_features):
        self.periods_agg = periods_agg
        self.agg_fun = agg_fun
        self.agg_fun_x_features = agg_fun_x_features
        
      if ensemble.validate_parameters(dt_units, transform,
                                      periods, periods_trig, holidays_df,
                                      models, consensus_method, pred_level):
        self.dt_units = dt_units
        self.transform = transform
        self.periods = periods
        self.periods_trig = periods_trig
        self.holidays_df = holidays_df
        self.models = models
        self.consensus_method = consensus_method
        self.pred_level = pred_level


        # Set during fitting or by other methods
        self.training = {'dt_span':None,
                         'history':None,
                         'aggregated':None,
                         'transformed':None,
                         'x_features':{}}
        self.ensemble_model_artifacts = {'coefficients':{},
                                         'summary':{},
                                         'residuals':{},
                                         'attributions':{}}
        self.forecast = {'dt_span':None,
                         'ensemble':{},
                         'consensus':None,
                         'disaggregated':{},
                         'x_features':{}}
        self.disaggregation_artifacts = {'coefficients':{},
                                         'residuals':{},
                                         'summaries':{}}
        self.validation = {'ensemble':{},
                           'consensus':None,
                           'disaggregated':{}}


  @staticmethod
  def _validate_df(df, dt_span=None, df_name='df'):
    """Validate a dataframe: ensure it contains a datetime column dt and covers
     the required dt span.
  
    Parameters
    ----------
    df: pd dataframe to validate
    dt_span: dict containing the begin_dt and end_dt that the df should span
    df_name: string name of dataframe to make any raised exceptions clearer
    
    Returns
    ----------
    Boolean indicating the dataframe is valid
    """
    
    if type(df) is not pd.core.frame.DataFrame:
      raise Exception(df_name + ' must be a pandas dataframe.')
    
    if 'dt' not in df.columns:
      raise Exception(df_name + ' must have column "dt"')
    
    if search('timestamp', str(type(df['dt'].iloc[0])).lower()) is None:
      raise Exception(df_name + ' column "dt" must be a timestamp.')
     
    if dt_span is not None:
      if min(df.dt) > dt_span['begin_dt'] or max(df.dt) < dt_span['end_dt']:
        raise Exception(df_name + ' does not span ' + 
                        str(dt_span['begin_dt']) + ' to ' +
                        str(dt_span['end_dt']))
    
    return True
     
    
  def set_training(self, df, x_features=None,
                   training_begin_dt=None, training_end_dt=None):
    """Set the training data: crop the df to the training dts and aggregate
     if directed to.
    
    Parameters
    ----------
    df: pd dataframe containing columns for dt datetime and actual, the 
        timeseries being forecasted
    x_features: pd dataframe containing columns for dt datetime and any 
        external regressors to be used in fitting the ensemble
    training_begin_dt: the desired begin dt of the training history
    training_end_dt: the desired end dt of the training history 
            
    Returns
    ----------
    A clairvoyant object updated with the desired training history set

    """
    
    self._validate_df(df)
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values(by='dt')
    if training_begin_dt is None: 
      training_begin_dt = min(df.dt)
    
    # By default set the training end date to 80
    if training_end_dt is None:
      training_end_dt = df.dt[np.ceil(df.shape[0] * 0.8)]
    
    training_dts = _get_begin_end_dts(training_begin_dt, training_end_dt, 
                                      self.dt_units, 
                                      fix_dt_option='begin_later',
                                      need_integer_period=
                                      (len(self.periods_agg) > 0),
                                      period=max([1] + self.periods_agg))
                                      
    self._validate_df(df, training_dts)
    self.training['dt_span'] = training_dts
    
    training_mask = ((df['dt'] >= training_dts['begin_dt']) & 
                     (df['dt'] <= training_dts['end_dt']))
    self.training['history'] = df.loc[training_mask]
    if 'actual' not in df.columns:
      raise Exception('df must contain column "actual".')
    
    aggregated_transformed = _aggregate_and_transform(
        self.training['history'], self.periods_agg, self.agg_fun, 
        transform=self.transform)
    
    self.training['aggregated'] = aggregated_transformed['aggregated']
    self.training['transformed'] = aggregated_transformed['transformed']
    
    if x_features is not None:
      self._validate_df(x_features, training_dts, 'x_features')
      cols_x_agg = x_features.columns[x_features.columns != 'dt']
      
      x_training_mask = ((x_features['dt'] >= training_dts['begin_dt']) &
                         (x_features['dt'] <= training_dts['end_dt']))
      self.training['x_features']['history'] = x_features.loc[x_training_mask]
    
      x_aggregated = _aggregate_and_transform(
          self.training['x_features']['history'],
          self.periods_agg,
          agg_fun=self.agg_fun_x_features,
          cols_agg=cols_x_agg,
          transform='none')
    
      self.training['x_features']['aggregated'] = x_aggregated['aggregated']
      self.training['x_features']['ensemble'] = x_aggregated['transformed']
      
      self.periods = [int(period / max(self.periods_agg + [1]))
                      for period in self.periods]
      self.periods_trig = [period / max(self.periods_agg + [1])
                           for period in self.periods_trig]
      
    else:
      self.training['x_features'] = None
   
    return self


  def set_forecast_x_features(self, x_features, fcst_dt_span):
    """Set the x features for forecasting.
    
    Parameters
    ----------
    x_features: pd dataframe containing columns for dt datetime and any 
        external regressors to be used in fitting the ensemble
    fcst_dt_span: dict containing the begin_dt and end_dt for the forecast
        
    Returns
    ----------
    An updated clairvoyant object with the x features for the forecast period
     set

    """   
    
    if x_features is not None:
      
      self._validate_df(x_features, fcst_dt_span, 'x_features')
      cols_x_agg = x_features.columns[x_features.columns != 'dt']
    
      x_forecast_mask = ((x_features['dt'] >= fcst_dt_span['begin_dt']) &
                         (x_features['dt'] <= fcst_dt_span['end_dt']))
      x_features_crop = x_features.loc[x_forecast_mask]
    
      x_aggregated = _aggregate_and_transform(
          x_features_crop,
          self.periods_agg,
          agg_fun=self.agg_fun_x_features,
          cols_agg=cols_x_agg,
          transform='none')
    
      self.forecast['x_features']['unaggregated'] = x_features_crop
      self.forecast['x_features']['aggregated'] = x_aggregated['aggregated']
      self.forecast['x_features']['ensemble'] = x_aggregated['transformed']
    
    else:
      self.forecast['x_features'] = None
        
    return self    


  @staticmethod
  def get_consensus_forecast(ensemble_df, consensus_method, transform='none'):
    """Get the consensus of the ensemble of forecasts using the specified 
        consensus method.
  
    Parameters
    ----------
    ensemble_df: pd dataframe containing all of the models' forecasts
    consensus_method: function to apply to get a consensus forecast at each dt
    df_name: string name of dataframe to make any raised exceptions clearer
    
    Returns
    ----------
    pd dataframe with columns for dt, the consensus forecast, and the lower and
       upper prediction interval limits
    """
    
    consensus_df = ensemble_df.groupby('dt').aggregate(
        {'forecast':consensus_method, 'forecast_lower':consensus_method,
         'forecast_upper':consensus_method})
    consensus_df.reset_index(inplace=True)
    consensus_df = _back_transform_df(consensus_df, transform)
    return consensus_df

  @staticmethod
  def get_consensus_attribution(ensemble_att_df, consensus_method,
                              transform='none'):
    """Get the consensus of the ensemble of forecasts using the specified 
        consensus method.

    Parameters
    ----------
    ensemble_att_df: pd dataframe containing all of the models' attributions
    consensus_method: function to apply to get a consensus forecast at each dt
    df_name: string name of dataframe to make any raised exceptions clearer
  
    Returns
    ----------
    pd dataframe with columns for dt, the consensus forecast, and the lower and
       upper prediction interval limits
    """
  
    consensus_dict = {}
    cols_att = ensemble_att_df.columns[ensemble_att_df.columns != 'dt']
    for att_col in cols_att:
      consensus_dict[att_col] = consensus_method
  
    consensus_df = ensemble_att_df.groupby('dt').aggregate(consensus_dict)
    
    consensus_df.reset_index(inplace=True)
    consensus_df = _back_transform_df(consensus_df, transform, 
                                      cols_transform=cols_att)
    return consensus_df
   

  def fit_ensemble(self, df, x_features=None,
                   training_begin_dt=None, training_end_dt=None,
                   forecast_end_dt=None,
                   x_cols_seasonal_interactions=[]):
    """Fit the specified ensemble of models and generate forecasts. Find 
      consensus forecast
    
    Parameters
    ----------
    df: pd dataframe containing columns for dt datetime and actual, the 
        timeseries being forecasted
    x_features: pd dataframe containing columns for dt datetime and any 
        external regressors to be used in fitting the ensemble
    training_begin_dt: the desired begin dt of the training history
    training_end_dt: the desired end dt of the training history
    forecast_end_dt: the desired end dt of the forecast period 
        
    Returns
    ----------
    A clairvoyant object updated with the desired validation set and set the 
      aggregated x_features to be used in the ensemble forecast

    """
    if self.training['dt_span'] is None or self.training['history'] is None:
      self = self.set_training(df, x_features, training_begin_dt,
                               training_end_dt)
      
    if self.forecast['dt_span'] is None:
      forecast_begin_dt = (self.training['dt_span']['end_dt'] + 
                           _datetime_delta(1, self.dt_units))
      self._validate_df(df)
      df['dt'] = pd.to_datetime(df['dt'])
      df = df.sort_values(by='dt')
      if forecast_end_dt is None:
        forecast_end_dt = max([max(df.dt), forecast_begin_dt + 
                               _datetime_delta(max(self.periods_agg + [2]) 
                                               * max(self.periods + [1]), 
                                               self.dt_units)])

      self.forecast['dt_span'] = _get_begin_end_dts(
          forecast_begin_dt, forecast_end_dt, self.dt_units,
          fix_dt_option='end_later', need_integer_period=
          (max(self.periods_agg + [1]) > 1), 
          period=max([1] + self.periods_agg))
       
    if x_features is not None:
      self = self.set_forecast_x_features(x_features,
                                          self.forecast['dt_span'])
    ensemble_df = pd.DataFrame() 
    ensemble_att_df = pd.DataFrame()
    if self.training['x_features'] is None:
      x_features_training = None
      x_features_forecast = None
    else:
      x_features_training = self.training['x_features'][
              'ensemble'].set_index('dt')
      x_features_forecast = self.forecast['x_features'][
              'ensemble'].set_index('dt')
    
    for model in self.models:
      model_str = str(model.__name__)  
      model_rslts = model(
          self.training['transformed'], self.forecast['dt_span'],
          self.dt_units, self.periods, self.periods_agg, self.periods_trig,
          self.pred_level, self.transform, 
          x_features_training, 
          x_features_forecast,
          self.holidays_df,
          x_cols_seasonal_interactions)

      self.forecast['ensemble'][model_str] = (
          model_rslts['forecast'])
      self.ensemble_model_artifacts['coefficients'][model_str] = (
          model_rslts['coefficients'])
      self.ensemble_model_artifacts['summary'][model_str] = (
          model_rslts['summary'])
      self.ensemble_model_artifacts['residuals'][model_str] = (
          model_rslts['residuals'])
      self.ensemble_model_artifacts['attributions'][model_str] = (
          model_rslts['attributions'])
      model_df = model_rslts['transformed_forecast']
      model_df['model'] = model_str
      ensemble_df = pd.concat([ensemble_df, model_df], ignore_index=True)
      
      if model_rslts['attributions'] is not None:
        ensemble_att_df = pd.concat([ensemble_att_df,
                                     model_rslts['attributions']],
                                    ignore_index=True)
      
    self.forecast['consensus'] = self.get_consensus_forecast(
        ensemble_df, self.consensus_method, self.transform)  
    self.ensemble_model_artifacts['attributions']['consensus'] = (
        self.get_consensus_attribution(
            ensemble_att_df, self.consensus_method, self.transform))                                                 

    return self


  def disaggregate_forecasts(self,
                             x_features_col_subset=[],
                             x_cols_seasonal_interactions=[]):
    """Disaggregate the consensus forecast to the original datetime units
    
    Parameters
    ----------
    x_features_col_subset: a list of the x_features pd dataframe columns to use
      in modelling the disaggregation proportions
    x_cols_seasonal_interactions: a list of the x_features pd dataframe columns
      to interact with the seasonal periods that were aggregated in modelling
      the disaggregation proportions
        
    Returns
    ----------
    A clairvoyant object updated with the disaggregated forecast

    """
    
    if self.training['aggregated'] is {}:
      raise Exception('Histories are not aggregated')
      
    if len(self.periods_agg) == 0 or max(self.periods_agg) <= 1:
      raise Exception('No periods are aggregated over')
    
    if self.forecast['consensus'] is None:
      raise Exception('No consensus forecast found') 
      
    if x_features_col_subset is not None and len(x_features_col_subset) > 0:
      if  not set(x_features_col_subset).issubset(
                  set(self.training['x_features']['history'].columns)):
        raise Exception('x_features_col_subset must be a subset of ' +
                        'original x_features columns')
        
      x_reg = self.training['x_features']
      x_future = self.forecast['x_features']
        
      if not set(x_cols_seasonal_interactions).issubset(
                 set(x_features_col_subset)):
        raise Exception('x_cols_seasonal_interactions must be a subset of ' +
                        'x_features_col_subset')
    else:
      x_reg = None
      x_future = None
        
    self.periods_agg.sort()
    
    periods_disagg = [1] + self.periods_agg[:-1]
    periods_disagg.sort(reverse=True)
    
    periods_agg = self.periods_agg
    periods_disagg.sort(reverse=True)
    
    period_str = 'period' + str(max(periods_agg))
    aggregated_history = self.training['aggregated'][period_str]
    
    aggregated_forecast = self.forecast['consensus']
    
    for period_agg, period_disagg in zip(periods_agg, periods_disagg):
        
      period_str_d = 'period' + str(period_disagg)
      
      if period_disagg == 1:
        history = self.training['history']
        if x_reg is not None and x_future is not None and len(
                x_features_col_subset) > 0:
          x_reg_d = x_reg['history'][x_features_col_subset]
          x_future_d = x_future['unaggregated'][x_features_col_subset]
          
        else:
          x_reg_d = None
          x_future_d = None
          
      else:
        history = self.training['aggregated'][period_str_d]
        if x_reg is not None and x_future is not None and len(
                x_features_col_subset) > 0:
          x_reg_d = x_reg['aggregated'][period_str_d][x_features_col_subset]
          x_future_d = x_future['aggregated'][period_str_d][
              x_features_col_subset]
        else:
          x_reg_d = None
          x_future_d = None
    
      disagg_object = disaggregate_forecast(
          history, aggregated_history, aggregated_forecast,
          self.dt_units, period_agg, period_disagg,
          x_reg_d, x_future_d, x_cols_seasonal_interactions,
          self.pred_level, self.agg_fun)
      
      self.forecast['disaggregated'][period_str_d] = (
          disagg_object['disaggregated'])
      
      self.disaggregation_artifacts['coefficients'] = (
          disagg_object['coefficients'])
      self.disaggregation_artifacts['summary'] = (
          disagg_object['summary'])
      self.disaggregation_artifacts['residuals'] = (
          disagg_object['residuals'])
      
      aggregated_forecast = self.forecast['disaggregated'][period_str_d]
      aggregated_history = history
       

    return self


  def get_out_of_time_validation(self, df):
    """Set the validation data: crop the df to the forecast dts and aggregate
     if directed to and then compute validation statistics on the available
     out of time validation data (that spans all or part of the forecast 
     period).

    Parameters
    ----------
    df: pd dataframe containing columns for dt datetime and actual, the 
        timeseries being forecasted
    x_features: pd dataframe containing columns for dt datetime and any 
        external regressors to be used in fitting the ensemble
    forecast_end_dt: the desired end dt of the forecast period 
        
    Returns
    ----------
    A clairvoyant object updated with the desired validation set and set the 
      aggregated x_features to be used in the ensemble forecast

    """
    self._validate_df(df)
    df = df.sort_values('dt')
    
    if self.forecast['dt_span'] is None:
      raise Exception('Forecast period not set.')
      
    
    validation_mask = ((df['dt'] >= self.forecast['dt_span']['begin_dt']) &
                       (df['dt'] <= self.forecast['dt_span']['end_dt']))
    validation_df = df.loc[validation_mask]

    aggregated = _aggregate_and_transform(
        validation_df, self.periods_agg, self.agg_fun, 
        transform='none')
    self.validation['consensus'] = aggregated['transformed']
    
    if self.forecast['ensemble'] is {}:
      raise Exception('No forecast ensemble found')
      
    for model in self.models:
      model_str = str(model.__name__)
      model_df = pd.concat(
          [self.validation['consensus'].drop(
              ['dt'], axis=1).reset_index(drop=True), 
           self.forecast['ensemble'][model_str].reset_index(drop=True)],
          axis=1)
      model_df.dropna()
      
      model_df['error'] = ((model_df['forecast'] - model_df['actual'])
                           / model_df['actual'])
      model_df['abs_error'] = np.abs(model_df['error'])
      
      model_df['pred_int_coverage'] = ((model_df['actual'] 
                                        >= model_df['forecast_lower']) &
                                       (model_df['actual']
                                       <= model_df['forecast_upper']))
      
      self.validation['ensemble'][model_str] = model_df
    
    if self.forecast['consensus'] is None:
      raise Exception('No consensus forecast found.')

    consensus_df = pd.concat(
        [self.validation['consensus'].drop(
            ['dt'], axis=1).reset_index(drop=True), 
         self.forecast['consensus'].reset_index(drop=True)],
        axis=1)

    consensus_df.dropna()
    
    consensus_df['error'] = ((consensus_df['forecast'] 
                              - consensus_df['actual'])
                           / consensus_df['actual'])
    consensus_df['abs_error'] = np.abs(consensus_df['error'])
    
    consensus_df['pred_int_coverage'] = ((consensus_df['actual'] 
                                        >= consensus_df['forecast_lower']) &
                                       (consensus_df['actual']
                                        <= consensus_df['forecast_upper']))
    
    self.validation['consensus'] = consensus_df
    
    if self.forecast['disaggregated'] != {}:
      disagg_fcst = self.forecast['disaggregated']['period1']
      history_mask = (df.dt >= min(disagg_fcst.dt))
      disagg_consensus_df = pd.concat(
          [df.drop(['dt'], axis=1).loc[history_mask,:].reset_index(
              drop=True),  disagg_fcst.reset_index(drop=True)],
          axis=1)

      disagg_consensus_df.dropna()
    
      disagg_consensus_df['error'] = (( disagg_consensus_df['forecast'] 
                                      -  disagg_consensus_df['actual'])
                                      /  disagg_consensus_df['actual'])
      disagg_consensus_df['abs_error'] = np.abs(disagg_consensus_df['error'])
    
      disagg_consensus_df['pred_int_coverage'] = ((
          disagg_consensus_df['actual'] >=  
          disagg_consensus_df['forecast_lower']) & (
              disagg_consensus_df['actual'] <=  
              disagg_consensus_df['forecast_upper']))
    
      self.validation['disaggregated']['period1'] = disagg_consensus_df
    
    return self

