# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

def validate_parameters(periods_agg=[7],
                        agg_fun=['sum'],
                        agg_fun_x_features=['sum']):
  """Function to validate aggregation parameters.
   Parameters
  ----------
  periods_agg: list of periods to aggregate over
  agg_fun: function to apply to actuals when aggregating periods
  agg_fun_x_features: list of functions to apply to x_features when 
    aggregating periods
  Returns
  -------
  Boolean indicated validated parameters.""" 
      
  if len([period for period in periods_agg if 
           period - np.round(period) == 0]) != len(periods_agg):
    raise Exception("Aggregated periods must be positive-integer valued")
        
  if len([period1 for period1 in periods_agg if 
          all(period1 % period2 == 0 for period2 in periods_agg if
              period1 > period2)]) != len(periods_agg):
     raise Exception("Aggregated periods must be positive-integer " +
                     "valued multiples of smaller values.")

  if (all(agg_fun_i not in ['sum', 'min', 'max', 'avg', 'median'] for
          agg_fun_i in agg_fun)):
    raise Exception("Aggregation functions must be sum, min, max, " +
                    "avg or median.")

  if (all(agg_fun_i not in ['sum', 'min', 'max', 'avg', 'median'] for
          agg_fun_i in agg_fun_x_features)):
    raise Exception("X feature aggregation functions must be sum, " +
                    "min, max, avg or median.")

  return True


def aggregate_to_longest(history,
                         periods_agg=[7],
                         agg_fun=['sum'],
                         cols_agg=['actual']):
  """Function to aggregate over seasonal periods.
   Parameters
  ----------
  history: pandas dataframe containing dt and actual
  periods_agg: list of periods to aggregate over
  agg_fun: function(s) to apply to columns being aggregated
  cols_agg: columns of history to aggregate
  Returns
  -------
  Named dict containing aggregated histories as pandas dataframes for each
  period aggregated over."""         
  
  if len(agg_fun) != len(cols_agg):
    agg_fun = agg_fun * len(cols_agg)
    agg_fun = agg_fun[:len(cols_agg)]
 
  periods_agg.sort()
  unaggregated_history = history.copy()
 
  aggregated_histories = {}
  period_m1 = 1
  
  for period in periods_agg:
    period_m = period / period_m1

    period_str = 'period' + str(period)
    
    # Create grouping variable for aggregation
    unaggregated_history['p_n'] = np.repeat(
        range(int(np.ceil(unaggregated_history.shape[0] / period_m))),
        period_m)[range(unaggregated_history.shape[0])]
    
    # Use pandas groupby instead of SQL
    agg_dict = {'dt': 'max'}
    for fun, col in zip(agg_fun, cols_agg):
      if fun == 'sum':
        agg_dict[col] = 'sum'
      elif fun == 'avg':
        agg_dict[col] = 'mean'
      elif fun == 'min':
        agg_dict[col] = 'min'
      elif fun == 'max':
        agg_dict[col] = 'max'
      elif fun == 'median':
        agg_dict[col] = 'median'
    
    # Group and aggregate
    rslt = unaggregated_history.groupby('p_n').agg(agg_dict).reset_index()
    
    # Filter to complete periods only
    rslt = rslt[rslt['p_n'].map(unaggregated_history.groupby('p_n').size()) == period_m]
    aggregated_history = rslt.drop(['p_n'], axis=1)
    aggregated_history['dt'] = pd.to_datetime(aggregated_history['dt'])
    
    period_str = 'period' + str(period)
    aggregated_histories[period_str] = aggregated_history.copy()
    unaggregated_history = aggregated_history.copy()
    period_m1 = period
    
  return aggregated_histories
