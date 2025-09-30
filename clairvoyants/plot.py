# -*- coding: utf-8 -*-
# Author Amy Richardson Fricke

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from clairvoyants.utilities import _datetime_delta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import norm

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def plot_ensemble(e, title=None, ylabel=''):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """
  if e.forecast['consensus'] is None:
    raise Exception('No consensus forecast found.')
    
  if e.forecast['ensemble'] is {}:
    raise Exception('No ensemble forecast found.')
   
  if title is None:
    title = 'Training and forecast model ensemble results'
  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))
  
  if len(e.periods_agg) > 0 and max(e.periods_agg) > 1:
    
    agg_str = 'period' + str(max(e.periods_agg))
    ep = plt.plot(e.training['aggregated'][agg_str].dt,
                  e.training['aggregated'][agg_str].actual,
                  label='training', linewidth=4)
    
  else:
    ep = plt.plot(e.training['history'].dt,
                  e.training['history'].actual, 
                  label='training', linewidth=4)
     
  for model in e.forecast['ensemble']:
    
    ep = plt.plot(e.forecast['ensemble'][model].dt,
                  e.forecast['ensemble'][model].forecast, 
                  label=model, linewidth=2)
    

    
  ep = plt.plot(e.forecast['consensus'].dt,
                e.forecast['consensus'].forecast,
                label='consensus forecast', linewidth=4, c='indianred')

     
  plt.legend(loc='upper left', ncol=4)
  plt.grid()
  
  return ep


def plot_ensemble_error(e, title=None, ylabel=''):
  """Produce a plot of the out of sample error of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """
  if e.validation['consensus'] is None:
    raise Exception('No consensus forecast validation found.')
    
  if e.validation['ensemble'] is {}:
    raise Exception('No ensemble forecast validation found.')
   
  if title is None:
    title = 'Training and forecast model ensemble results'
  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  
  ax.yaxis.set_major_formatter(mtick.PercentFormatter())
  ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
  
  for model in e.validation['ensemble']:

    eep = plt.plot(e.validation['ensemble'][model].dt,
                   e.validation['ensemble'][model].error * 100,
                   label=model, linewidth=2)
  
  eep = plt.plot(e.validation['consensus'].dt,
                 e.validation['consensus'].error * 100,
                 label='consensus error', linewidth=4, c='indianred')
  
  plt.axhline(y=0, color='grey', linestyle='--', linewidth=2)
  plt.legend(loc='upper left', numpoints=1, ncol=4)
  plt.grid()
  
  return eep


def plot_training_actuals_forecast(e, title=None, ylabel='', 
                                   include_pred_int=False):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """
  if e.forecast['consensus'] is None:
    raise Exception('No forecast found.')
       
  if title is None and e.validation['consensus'] is not None:
    title = 'Training, forecast and actuals'

  if title is None and e.validation['consensus'] is None:
    title = 'Training and forecast'
  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))

  
  if len(e.periods_agg) > 0 and max(e.periods_agg) > 1:
    
    agg_str = 'period' + str(max(e.periods_agg))
    
    fp = plt.errorbar(e.training['aggregated'][agg_str].dt,
                      e.training['aggregated'][agg_str].actual, yerr=None,
                      label='training', linewidth=4)
    history_len = e.training['aggregated'][agg_str].shape[0]
    
  else:
    fp = plt.errorbar(e.training['history'].dt,
                      e.training['history'].actual, yerr=None,
                      label='training', linewidth=2)
    history_len = e.training['history'].shape[0]
    
  if include_pred_int:
    pred_int = [(e.forecast['consensus'].forecast
                 - e.forecast['consensus'].forecast_lower),
                (e.forecast['consensus'].forecast_upper
                 - e.forecast['consensus'].forecast)]

  else:
    pred_int = None

  total_len = history_len + e.forecast['consensus'].shape[0]    
  fp = plt.errorbar(e.forecast['consensus'].dt,
                    e.forecast['consensus'].forecast, yerr=pred_int,
                    label='forecast', 
                    linewidth=2 + 2 * int(total_len < 400),
                    c='indianred')
  
  
  if include_pred_int:
    fp[-1][0].set_linestyle(':')
    fp[-1][0].set_linewidth(1 + int(total_len < 400))
    fp[-1][0].set_alpha(.5 + .25 * int(total_len < 400))
 
  if (e.validation['consensus'] is not None and 
      len(e.validation['consensus']) > 0):
    fp = plt.errorbar(e.validation['consensus'].dt,
                      e.validation['consensus'].actual, yerr=None,
                      label='actuals', c='mediumseagreen', 
                      linewidth=2 + 2 * int(total_len < 400), alpha=0.7)

  plt.legend(loc='upper left', ncol=3)
  plt.grid()
  
  return fp


def plot_forecast_scenarios(re, ces, ref_name=None, title=None, ylabel='',
                            training_display_start=None):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """
  if re.forecast['consensus'] is None:
    raise Exception('No reference forecast found.')
    
  if ces is None or ces is {}:
    raise Exception('No comparison forecasts found.')
    
  if ref_name is None:
    ref_name = 'reference forecast'
       
  if title is None:
    title = 'Forecast comparison across ' + len(ces) + ' scenarios'
    
  if training_display_start is None:
    training_display_start = (max(re.training['history'].dt) -
                              _datetime_delta(2, 'Y'))
    
  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))

  if len(re.periods_agg) > 0 and max(re.periods_agg) > 1:
    
    agg_str = 'period' + str(max(re.periods_agg))
    
    training_mask = (
        re.training['aggregated'][agg_str].dt >= training_display_start)
    
    cp = plt.plot(re.training['aggregated'][agg_str].dt.loc[training_mask],
                  re.training['aggregated'][agg_str].actual.loc[training_mask], 
                  label='training', linewidth=4)
    
  else:
    training_mask = (
        re.training['history'].dt >= training_display_start)
    cp = plt.plot(re.training['history'].dt.loc[training_mask],
                  re.training['history'].actual.loc[training_mask],
                  label='training', linewidth=4)

  for ckey in ces.keys():
    cp = plt.plot(ces[ckey].forecast['consensus'].dt,
                  ces[ckey].forecast['consensus'].forecast, 
                  label=ckey, linewidth=2)
         
  cp = plt.plot(re.forecast['consensus'].dt,
                re.forecast['consensus'].forecast, 
                label=ref_name, linewidth=4, c='cyan')
  
  
  plt.legend(loc='upper left', ncol=3)
  plt.grid()
  
  return cp


def plot_historical_actuals_forecast(e, title=None, ylabel='', 
                                     include_pred_int=False,
                                     years_prior_include=2,
                                     forecast_display_start=None,
                                     e2=None):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """
  if e.forecast['consensus'] is None:
    raise Exception('No forecast found.')
       
  if title is None and e.validation['consensus'] is not None:
    title = 'Training, forecast and actuals'

  if title is None and e.validation['consensus'] is None:
    title = 'Training and forecast'
  
  fig, ax = plt.subplots(figsize=(13, 11))
  fig.suptitle(title, fontsize=24)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  plt.xticks(rotation = 30)
  
  ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))
  
  if forecast_display_start is None:
    forecast_display_start = min(e.forecast['consensus'].dt)
    
  forecast_mask = (e.forecast['consensus'].dt >= forecast_display_start)
  forecast_len = forecast_mask.sum()
  
  max_vals = []
  
  for yp in list(range(1, years_prior_include + 1)):
    
    if len(e.periods_agg) > 0 and max(e.periods_agg) > 1:
      
      agg_str = 'period' + str(max(e.periods_agg)) 
      range_train_yp = {'min':(forecast_display_start - 
                                _datetime_delta(yp, 'Y') + 
                                _datetime_delta(yp, 'D')),
                        'max':(max(e.forecast['consensus'].dt) - 
                                _datetime_delta(yp, 'Y') + 
                                _datetime_delta(yp, 'D'))}
      
      training_mask = (
          (e.training['aggregated'][agg_str].dt >= range_train_yp['min']) &
          (e.training['aggregated'][agg_str].dt <= range_train_yp['max']))

      train_len = training_mask.sum()
      fp = plt.plot(e.forecast['consensus'].dt.loc[forecast_mask][:train_len],
                    e.training['aggregated'][agg_str].actual.loc[
                        training_mask][:forecast_len],
                    label='actuals ' + str(int(yp)) + 'YA', 
                    linewidth=4)
      history_len = e.training['aggregated'][agg_str].shape[0]
      max_vals = max_vals + [max(
          e.training['aggregated'][agg_str].actual.loc[
          training_mask][:forecast_len])]

    
    else:
      range_train_yp = {'min':(forecast_display_start - 
                                _datetime_delta(yp, 'Y')),
                        'max':(max(e.forecast['consensus'].dt) - 
                                _datetime_delta(yp, 'Y'))}
      training_mask = (
          (e.training['history'].dt >= range_train_yp['min']) &
          (e.training['history'].dt <= range_train_yp['max']))
      
      fp = plt.plot(e.forecast['consensus'].dt.loc[forecast_mask],
                    e.training['history'].actual.loc[training_mask],
                    label='actuals ' + str(int(yp)) + 'YA', linewidth=2)
      history_len = e.training['history'].shape[0]
      max_vals = max_vals + [max(
          e.training['history'].actual.loc[training_mask])]


  total_len = history_len + e.forecast['consensus'].shape[0]    
  fp = plt.plot(e.forecast['consensus'].dt.loc[forecast_mask],
                e.forecast['consensus'].forecast.loc[forecast_mask],
                label='forecast', 
                linewidth=2 + 2 * int(total_len < 400),
                c='indianred')
  
  max_vals = max_vals + [max(
      e.forecast['consensus'].forecast.loc[forecast_mask])]

  if include_pred_int:
    fp = plt.fill_between(e.forecast['consensus'].dt.loc[forecast_mask], 
                          e.forecast['consensus'].forecast_lower.loc[
                              forecast_mask],
                          e.forecast['consensus'].forecast_upper.loc[
                              forecast_mask],
                          color='indianred', alpha=0.3,
                          label=str(round(
                              e.pred_level * 100)) + '% prediction band')
    max_vals = max_vals + [max(e.forecast['consensus'].forecast_upper.loc[
        forecast_mask])]
     
  
  if (e.validation['consensus'] is not None and 
      len(e.validation['consensus']) > 0):
    fp = plt.plot(e.validation['consensus'].dt.loc[forecast_mask],
                  e.validation['consensus'].actual.loc[forecast_mask], 
                  label='actuals', c='mediumseagreen', 
                  linewidth=2 + 2 * int(total_len < 400))
    
    max_vals = max_vals + [max(
        e.validation['consensus'].actual.loc[forecast_mask])]

    
  if (e2 is not None and 
      len(e.forecast['consensus'].dt) > 0):
    forecast_mask2 = (e2.forecast['consensus'].dt >= forecast_display_start)
    fp = plt.plot(e2.forecast['consensus'].dt.loc[forecast_mask2],
                  e2.forecast['consensus'].forecast.loc[forecast_mask2],
                  label='latest forecast', 
                  linewidth=2 + 2 * int(total_len < 400),
                  c='purple')
    max_vals = max_vals + [max(
        e2.forecast['consensus'].forecast.loc[forecast_mask2])]
    

  plt.ylim([0, 1.05 * max(max_vals)])
  plt.legend(loc='lower center', ncol=3, framealpha=0.05)
  plt.grid()
  
  return fp


def plot_attribution(a, a_display=None, title=None, ylabel='',
                     conf_level=0.95):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """

  a = a.reset_index(drop=True)
  if type(a) is not pd.core.frame.DataFrame:
    raise Exception('a, the attribution dataframe must be pandas.')

  if not set(['dt', 'actual']).issubset(set(a.columns)):
    raise Exception('a must contain columns "dt" and "actual"')
    
  if a_display is None:
    a_display = {}
    for col in list(set(a.columns) - set(['dt', 'total'])):
      a_display[col] = [col]
   
  if title is None:
    title = 'Model predicted attributions of various factors'

  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=13)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))

  dt_series = a.dt.copy()
  a_upper = a.actual.copy()
  a_lower = a.actual.copy()
  a_upper_neg = None
  a_lower_neg = None
  a_lower_neg_first = None

  
  for a_key in a_display.keys():
    a_series = a[a_display[a_key][0]]
    a_se = a[a_display[a_key][0] + '_var']
    
    for a_col in a_display[a_key][1:]:
      a_series += a[a_col]
      a_se += a[a_col + '_var']
    
    a_se = np.sqrt(a_se)
    
    a_pos = (a_series > 0)
    a_neg = (a_series < 0)
    
    if a_pos.sum() > 0:
      
      a_lower.loc[a_pos] = a_lower.loc[a_pos] - a_series.loc[a_pos]
      a_nonneg = (a_series >= 0)
      if a_nonneg.sum() != len(a_series):
       cur_col = next(plt.gca()._get_lines.prop_cycler)['color']
       
      ap = plt.fill_between(dt_series, 
                            a_upper,
                            a_lower,
                            label=a_key + ' driven', alpha=0.5)
      
      if a_nonneg.sum() == len(a_series):                      
        ap = plt.errorbar(dt_series.loc[a_pos], 
                          a_upper.loc[a_pos], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_pos],
                          label=(a_key + ' ' + str(round(100 * conf_level))
                                + '% CI'),
                          linewidth=0)
      else: 
       ap = plt.errorbar(dt_series.loc[a_pos], 
                         a_upper.loc[a_pos], 
                         yerr=np.abs(norm.ppf(
                             (1 - conf_level) / 2 )) * a_se[a_pos],
                         label=(a_key + ' ' + str(round(100 * conf_level))
                               + '% CI'),
                         linewidth=0,
                         ecolor=cur_col)

      ap[-1][0].set_linestyle(':')
      ap[-1][0].set_linewidth(2.5)
      
      
      a_upper = a_lower.copy()
 
    
    if a_neg.sum() > 0:

      if a_upper_neg is None and a_lower_neg is None:
        a_upper_neg = pd.Series([0] * len(a_series))
        a_lower_neg = pd.Series([0] * len(a_series))
        a_lower_neg.loc[a_neg] = a_series.loc[a_neg]
        a_lower_neg_first = a_lower_neg
        
      else:
        a_lower_neg.loc[a_neg] = a_lower_neg.loc[a_neg] + a_series.loc[a_neg]
        
      a_nonpos = (a_series <= 0)
      if a_nonpos.sum() == len(a_series):
        
        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              label=a_key + ' driven', alpha=0.5)
                            
        ap = plt.errorbar(dt_series.loc[a_neg], 
                          a_upper_neg.loc[a_neg], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_neg],
                          label=(a_key + ' ' + str(round(100 * conf_level))
                                + '% CI'),
                          linewidth=0)
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)
      
      else:

        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              color=cur_col, alpha=0.5)
                            
        ap = plt.errorbar(dt_series.loc[a_neg], 
                          a_upper_neg.loc[a_neg], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_neg],
                          ecolor=cur_col, linewidth=0)
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)
         

      
      a_upper_neg = a_lower_neg.copy()

    
  if a_lower_neg_first is None:
    a_lower = [0] * len(a.total)
  else:
    a_lower = a_lower_neg_first
 
  ap = plt.fill_between(dt_series, [0] * len(a.total), a_upper,
                        color='grey',
                        label = 'other/unknown', alpha=0.35)
  
     
  plt.legend(loc='upper left', ncol=2, framealpha=0.02)
  plt.grid()
  
  return ap

custom_color_map = ['tab:orange',  'tab:blue', 
                    'tab:red', 'darkcyan', 'tab:pink', 'darkslateblue', 
                    'goldenrod', 'tab:brown', 'tab:purple', 'tab:olive', 
                    'tab:grey', 'tab:cyan', 'coral']
def plot_attributions2(a, a_display=None, title=None, ylabel='',
                      conf_level=0.95):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """

  a = a.reset_index(drop=True)
  if type(a) is not pd.core.frame.DataFrame:
    raise Exception('a, the attribution dataframe must be pandas.')

  if not set(['dt', 'actual']).issubset(set(a.columns)):
    raise Exception('a must contain columns "dt" and "actual"')
    
  if a_display is None:
    a_display = {}
    for col in list(set(a.columns) - set(['dt', 'total'])):
      a_display[col] = [col]
   
  if title is None:
    title = 'Model predicted attributions of various factors'

  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=13)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  plt.xticks(rotation = 30)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))

  dt_series = a.dt.copy()
  a_upper = a.actual.copy()
  a_lower = a.actual.copy()
  a_upper_neg = None
  a_lower_neg = None
  a_lower_pos = None
  a_upper_pos = None

  i = -1
  for a_key in a_display.keys():
    i += 1
    cur_col = custom_color_map[i]
    a_series = a[a_display[a_key][0]]
    a_se = a[a_display[a_key][0] + '_var']
    
    for a_col in a_display[a_key][1:]:
      a_series += a[a_col]
      a_se += a[a_col + '_var']
    
    a_se = np.sqrt(a_se)
    
    a_pos = (a_series > 0)
    a_neg = (a_series < 0)
    
    if a_pos.sum() > 0:
      
      a_nonneg = (a_series >= 0)
      
      if a_nonneg.sum() == len(a_series):
        a_lower.loc[a_pos] = a_lower.loc[a_pos] - a_series.loc[a_pos]
          
        ap = plt.fill_between(dt_series, 
                              a_upper,
                              a_lower,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
      
        a_upper = a_lower.copy()
 
    if a_neg.sum() > 0:

      if a_upper_neg is None and a_lower_neg is None:
        a_upper_neg = pd.Series([0] * len(a_series))
        a_lower_neg = pd.Series([0] * len(a_series))
        a_lower_neg.loc[a_neg] = a_series.loc[a_neg]

        
      else:
        a_lower_neg.loc[a_neg] = a_lower_neg.loc[a_neg] + a_series.loc[a_neg]
        
        
      a_nonpos = (a_series <= 0)
      if a_nonpos.sum() == len(a_series):
        
        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
            
      
      else:
        
        if a_upper_pos is None and a_lower_pos is None:
          a_upper_pos = pd.Series([0] * len(a_series))
          a_lower_pos = pd.Series([0] * len(a_series))
          a_upper_pos.loc[a_pos] = a_series.loc[a_pos]
          
        else:
          a_upper_pos.loc[a_pos] = a_upper_pos.loc[a_pos] + a_series[a_pos]
         
        ap = plt.fill_between(dt_series, 
                              a_upper_pos, a_lower_pos,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
                            

        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              color=[cur_col], edgecolor=None,
                              alpha=0.5)
                            
        
        a_lower_pos = a_upper_pos.copy()

      
      a_upper_neg = a_lower_neg.copy()

  
  
  if a_upper_pos is None:
    a_lower = [0] * len(a.total)
  else: 
    a_lower = a_upper_pos

  
  ap = plt.fill_between(dt_series, a_lower, a_upper,
                        color=['grey'],
                        label = 'other/unknown', alpha=0.35)
  
     
  plt.legend(loc='upper left', ncol=2, framealpha=0.02)
  plt.grid()
  
  return ap

def plot_attributions(a, a_display=None, title=None, ylabel='',
                      conf_level=0.95):
  """Produce a plot of the ensemble forecasts

      
  Returns
  ----------
  plt object

  """

  a = a.reset_index(drop=True)
  if type(a) is not pd.core.frame.DataFrame:
    raise Exception('a, the attribution dataframe must be pandas.')

  if not set(['dt', 'actual']).issubset(set(a.columns)):
    raise Exception('a must contain columns "dt" and "actual"')
    
  if a_display is None:
    a_display = {}
    for col in list(set(a.columns) - set(['dt', 'total'])):
      a_display[col] = [col]
   
  if title is None:
    title = 'Model predicted attributions of various factors'

  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=13)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))


  max_pt = max(a.actual) * 1.1
  dt_series = a.dt.copy()
  a_upper = a.actual.copy()
  a_lower = a.actual.copy()
  a_upper_neg = None
  a_lower_neg = None
  a_lower_pos = None
  a_upper_pos = None

  i = -1
  for a_key in a_display.keys():
    i += 1
    cur_col = custom_color_map[i]
    a_series = a[a_display[a_key][0]]
    a_se = a[a_display[a_key][0] + '_var']
    
    for a_col in a_display[a_key][1:]:
      a_series += a[a_col]
      a_se += a[a_col + '_var']
    
    a_se = np.sqrt(a_se)
    
    a_pos = (a_series > 0)
    a_neg = (a_series < 0)
    
    if a_pos.sum() > 0:
      
      a_nonneg = (a_series >= 0)
      
      if a_nonneg.sum() == len(a_series):
        a_lower.loc[a_pos] = a_lower.loc[a_pos] - a_series.loc[a_pos]
          
        ap = plt.fill_between(dt_series, 
                              a_upper,
                              a_lower,
                              label=a_key + ' driven', 
                              color=[cur_col],
                              alpha=0.5, 
                              edgecolor=None)
                      
        ap = plt.errorbar(dt_series.loc[a_pos], 
                          a_upper.loc[a_pos], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_pos],
                          label=(a_key + ' ' + str(round(100 * conf_level))
                                + '% CI'),
                          linewidth=0,
                          ecolor=[cur_col],
                          edgecolor=None)
     
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)
      
        a_upper = a_lower.copy()
 
    
    if a_neg.sum() > 0:

      if a_upper_neg is None and a_lower_neg is None:
        a_upper_neg = pd.Series([0] * len(a_series))
        a_lower_neg = pd.Series([0] * len(a_series))
        a_lower_neg.loc[a_neg] = a_series.loc[a_neg]

        
      else:
        a_lower_neg.loc[a_neg] = a_lower_neg.loc[a_neg] + a_series.loc[a_neg]
        
        
      a_nonpos = (a_series <= 0)
      if a_nonpos.sum() == len(a_series):
        
        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
                            
        ap = plt.errorbar(dt_series.loc[a_neg], 
                          a_upper_neg.loc[a_neg], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_neg],
                          label=(a_key + ' ' + str(round(100 * conf_level))
                                + '% CI'),
                          linewidth=0,
                          ecolor=[cur_col], edgecolor=None)
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)
      
      else:
        
        
        if a_upper_pos is None and a_lower_pos is None:
          a_upper_pos = pd.Series([0] * len(a_series))
          a_lower_pos = pd.Series([0] * len(a_series))
          a_upper_pos.loc[a_pos] = a_series.loc[a_pos]
          
        else:
          a_upper_pos.loc[a_pos] = a_upper_pos.loc[a_pos] + a_series[a_pos]
         
        ap = plt.fill_between(dt_series, 
                              a_upper_pos, a_lower_pos,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
                            
        ap = plt.errorbar(dt_series.loc[a_pos], 
                          a_upper_pos.loc[a_pos], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_pos],
                          label=(a_key + ' ' + str(round(100 * conf_level))
                                + '% CI'), linewidth=0,
                          ecolor=[cur_col])
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)

        
        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              color=[cur_col], alpha=0.5, edgecolor=None)
                            
        ap = plt.errorbar(dt_series.loc[a_neg], 
                          a_lower_neg.loc[a_neg], 
                          yerr=np.abs(norm.ppf(
                              (1 - conf_level) / 2 )) * a_se[a_neg],
                          ecolor=[cur_col], linewidth=0)
        ap[-1][0].set_linestyle(':')
        ap[-1][0].set_linewidth(2.5)
        
        a_lower_pos = a_upper_pos.copy()

      
      a_upper_neg = a_lower_neg.copy()

    
  if a_upper_pos is None:
    a_lower = [0] * len(a.total)
  else:
    a_lower = a_upper_pos
 
  ap = plt.fill_between(dt_series, a_lower, a_upper,
                        color='grey',
                        label = 'other/unknown', alpha=0.35,
                        edgecolor=None)
  
     
  ap = plt.axhline(max_pt, linewidth=0)
  plt.legend(loc='upper left', ncol=2, framealpha=0.02)
  plt.grid()
  
  return ap

def plot_attributions3(a, a_display=None, title=None, ylabel='',
                      conf_level=0.95):
  """Produce a plot of the attributions

      
  Returns
  ----------
  plt object

  """

  a = a.reset_index(drop=True)
  if type(a) is not pd.core.frame.DataFrame:
    raise Exception('a, the attribution dataframe must be pandas.')

  if not set(['dt', 'actual']).issubset(set(a.columns)):
    raise Exception('a must contain columns "dt" and "actual"')
    
  if a_display is None:
    a_display = {}
    for col in list(set(a.columns) - set(['dt', 'total'])):
      a_display[col] = [col]
      print(a_display)
   
  if title is None:
    title = 'Model predicted attributions of various factors'

  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('datetime', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=13)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  plt.xticks(rotation = 30)
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))

  a.actual = a.actual + 110
  dt_series = a.dt.copy()
  a_upper = a.actual.copy()
  a_lower = a.actual.copy()
  a_upper_neg = None
  a_lower_neg = None
  a_lower_pos = None
  a_upper_pos = None

  i = -1
  for a_key in a_display.keys():
    i += 1
    cur_col = custom_color_map[i]
    a_series = a[a_display[a_key][0]]
    a_se = a[a_display[a_key][0] + '_var']
    
    for a_col in a_display[a_key][1:]:
      a_series += a[a_col]
      a_se += a[a_col + '_var']
    
    a_se = np.sqrt(a_se)
    
    a_pos = (a_series > 0)
    a_neg = (a_series < 0)
    
    if a_pos.sum() > 0:
      
      a_nonneg = (a_series >= 0)
      
      if a_nonneg.sum() == len(a_series):
        a_lower.loc[a_pos] = a_lower.loc[a_pos] - a_series.loc[a_pos]
          
        ap = plt.fill_between(dt_series, 
                              a_upper,
                              a_lower,
                              label=a_key, alpha=0.5,
                              color=[cur_col], edgecolor=None)
      
        a_upper = a_lower.copy()
 
    if a_neg.sum() > 0:

      if a_upper_neg is None and a_lower_neg is None:
        a_upper_neg = pd.Series([0] * len(a_series))
        a_lower_neg = pd.Series([0] * len(a_series))
        a_lower_neg.loc[a_neg] = a_series.loc[a_neg]

        
      else:
        a_lower_neg.loc[a_neg] = a_lower_neg.loc[a_neg] + a_series.loc[a_neg]
        
        
      a_nonpos = (a_series <= 0)
      if a_nonpos.sum() == len(a_series):
        
        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              label=a_key + ' driven', alpha=0.5,
                              color=[cur_col], edgecolor=None)
            
      
      else:
        
        if a_upper_pos is None and a_lower_pos is None:
          a_upper_pos = pd.Series([0] * len(a_series))
          a_lower_pos = pd.Series([0] * len(a_series))
          a_upper_pos.loc[a_pos] = a_series.loc[a_pos]
          
        else:
          a_upper_pos.loc[a_pos] = a_upper_pos.loc[a_pos] + a_series[a_pos]
         
        ap = plt.fill_between(dt_series, 
                              a_upper_pos, a_lower_pos,
                              label=a_key, alpha=0.5,
                              color=[cur_col], edgecolor=None)
                            

        ap = plt.fill_between(dt_series, 
                              a_upper_neg, a_lower_neg,
                              color=[cur_col], edgecolor=None,
                              alpha=0.5)
                            
        
        a_lower_pos = a_upper_pos.copy()

      
      a_upper_neg = a_lower_neg.copy()

  
  
  if a_upper_pos is None:
    a_lower = [0] * len(a.total)
  else: 
    a_lower = a_upper_pos

  
  ap = plt.fill_between(dt_series, a_lower, a_upper,
                        color=['grey'],
                        label = 'other/unknown', alpha=0.35)
  
     
  plt.legend(loc='upper left', ncol=2, framealpha=0.02)
  plt.grid()
  
  return ap


def plot_pre_post_actuals_forecast(e, title=None, ylabel='', 
                                   use_aggregated=False,
                                   include_pred_int=True,
                                   pre_period=14,
                                   post_period=30,
                                   dt_unit_str='Days'):
  """Produce a plot of the pre and post period using the forecast as a bench
    mark.

      
  Returns
  ----------
  plt object

  """
  if e.forecast['consensus'] is None:
    raise Exception('No forecast found.')
       
  if title is None and e.validation['consensus'] is not None:
    title = 'Pre and post analysis: actuals vs forecast'

  if title is None and e.validation['consensus'] is None:
    title = 'Training and forecast'
  
  fig, ax = plt.subplots(figsize=(15, 9))
  fig.suptitle(title, fontsize=24)
  plt.xlabel('date', fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.rc('legend', fontsize=18)
  plt.rc('ytick', labelsize=18)
  plt.rc('xtick', labelsize=18)
  plt.xticks(rotation = 30)
  
  ax.yaxis.set_major_formatter(FuncFormatter(human_format))
  #ax2 = ax.twiny()
  #ax2.set_xlabel(dt_unit_str + " since launch")
  #ax2.set_xticks(list(range(-1 * pre_period, post_period)))
   
  if (len(e.periods_agg) > 0 and max(e.periods_agg) > 1 and
      use_aggregated):
    
    agg_str = 'period' + str(max(e.periods_agg))
    pre_start = e.training['aggregated'][agg_str].shape[0] - pre_period
    fp = plt.plot(e.training['aggregated'][agg_str].dt.iloc[pre_start:],
                  e.training['aggregated'][agg_str].actual.iloc[pre_start:], 
                  label='actuals pre period', linewidth=4)
    history_len = e.training['aggregated'][agg_str].shape[0]
    
    if include_pred_int:
      fp = plt.fill_between(e.forecast['consensus'].dt.iloc[:post_period],
                            e.forecast['consensus'].forecast_lower.iloc[
                                :post_period],
                            e.forecast['consensus'].forecast_upper.iloc[
                                :post_period],
                            color='indianred',
                            label = str(round(e.pred_level * 100)) + 
                            '% prediction interval', alpha=0.2) 
      
    total_len = history_len + e.forecast['consensus'].shape[0]  
    fp = plt.plot(e.forecast['consensus'].dt.iloc[:post_period],
                  e.forecast['consensus'].forecast.iloc[:post_period], 
                  label='forecast', 
                  linewidth=2 + 2 * int(total_len < 400),
                  c='indianred')

      
    if (e.validation['consensus'] is not None and 
        len(e.validation['consensus']) > 0):
      fp = plt.plot(e.validation['consensus'].dt.iloc[:post_period],
                    e.validation['consensus'].actual.iloc[:post_period], 
                    label='actuals post period', c='mediumseagreen', 
                    linewidth=2 + 2 * int(total_len < 400), alpha=0.7)
      plt.axvline(x=min(e.forecast['consensus'].dt), color='grey',
                  linestyle='--', linewidth=5, label='launch')
    
  else:
    pre_start = e.training['history'].shape[0] - pre_period
    fp = plt.plot(e.training['history'].dt.iloc[pre_start:],
                  e.training['history'].actual.iloc[pre_start:],
                  label='actuals pre period', 
                  linewidth=2 + 2 * int(pre_period < 400))
    history_len = pre_period
    
    forecast_df = e.forecast['disaggregated']['period1']
    if include_pred_int:
      fp = plt.fill_between(forecast_df.dt.iloc[:post_period],
                            forecast_df.forecast_lower.iloc[:post_period],
                            forecast_df.forecast_upper.iloc[:post_period],
                            color='indianred',
                            label = str(round(e.pred_level * 100)) + 
                            '% prediction interval', alpha=0.2)
      
    total_len = history_len + e.forecast['consensus'].shape[0]  
    fp = plt.plot(forecast_df.dt.iloc[:post_period],
                  forecast_df.forecast.iloc[:post_period],
                  label='forecast', 
                  linewidth=2 + 2 * int(total_len < 400),
                  c='indianred')
    #ax.set_xticks(list(e.training['history'].dt.iloc[pre_start:]) + 
    #              list(forecast_df.dt.iloc[:post_period]))
    if (e.validation['disaggregated'] is not {} and 
        len(e.validation['disaggregated']) > 0):
      validation_df = e.validation['disaggregated']['period1']
      fp = plt.plot(validation_df.dt.iloc[:post_period],
                    validation_df.actual.iloc[:post_period], 
                    label='actuals post period', c='mediumseagreen', 
                    linewidth=2 + 2 * int(total_len < 400), alpha=0.7)
      
      plt.axvline(x=min(forecast_df.dt), color='grey', linestyle='--', 
                  linewidth=5, label='launch')

  plt.legend(loc='upper left', ncol=2)
  plt.grid()
  
  return fp

