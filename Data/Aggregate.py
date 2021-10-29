#!/usr/bin/env python
"""
Aggregate.py
Author: Adam N Morgan

Collection of aggregation functions
"""

import pandas as pd
import numpy as np


def replace_values(series, rule_dict):
    """Given a pandas series and a rule_dict with keys 'comparison_val',
      'operator', and 'replace_val', replace all series values that are
      <operator> <comparison_val> with <replace_val>.

    This can be used to set all values lower than some minimum value to that
      minimum value [e.g. set all values < 0.0 to 0.0] by using the dictonary
      {'operator':'<','comparison_val':0.0,'replace_val':0.0}.

    Similarly, setting all values larger than a maximum value of 100:
      {'operator':'>','comparison_val':100.0,'replace_val':100.0}.

    One can 'remove' outliers or invalid values by setting them to NaN.
      e.g. mark values of -1 as 'missing':
         {'operator':'==','comparison_val':-1,'replace_val':np.nan}
      e.g. mark values greater than 150 as 'missing':
         {'operator':'>','comparison_val':150,'replace_val':np.nan}

    N.B.: Removing or changing outlying points can greatly affect the
      calculations of the aggregate statistics, in particular the standard
      deviation and the mean (the median and median absolute deviation are more
      robust against outliers, so they should not change as much).  Be cautious
      about changing data unless you know the source of the problem you're
      looking to correct.  It is also important to keep records to any changes
      to the data that are made.
    """
    if rule_dict['operator'].strip() == '<':
        wheretrue = (series < rule_dict['comparison_val'])
    elif rule_dict['operator'].strip() == '<=':
        wheretrue = (series <= rule_dict['comparison_val'])
    elif rule_dict['operator'].strip() == '>':
        wheretrue = (series > rule_dict['comparison_val'])
    elif rule_dict['operator'].strip() == '>=':
        wheretrue = (series >= rule_dict['comparison_val'])
    elif rule_dict['operator'].strip() == '==':
        wheretrue = (series == rule_dict['comparison_val'])
    elif rule_dict['operator'].strip() == '!=':
        wheretrue = (series != rule_dict['comparison_val'])
    else:
        raise ValueError("Invalid operator in rule_dict!")

    print("  *** Replacing {0} values that are {1} {2} with {3}".format(
        len(series.loc[wheretrue]),
        rule_dict['operator'],
        rule_dict['comparison_val'],
        rule_dict['replace_val']))

    series.loc[wheretrue] = rule_dict['replace_val']

    return series


class qSeries(object):
    '''Series of data upon which aggregate functions can be applied.

    TODO: Better documentation for this class!

    In the meantime, see example functions below.
    '''
    def __init__(self, values, errors=None, t_deltas=None, index=None,
                 rule_list=[], weight_fn=None):
        if weight_fn is None:
            weight_fn = Aggregate.no_weight
        self.df = pd.DataFrame(values, index=index, columns=['vals_raw'])
        self.vals_raw = self.df['vals_raw']
        self.N_raw = len(self.vals_raw)
        # replace certain values according to list of rule_dicts
        self.rule_list = rule_list
        self.df.loc[:, 'vals'] = self.df['vals_raw']
        for rule_dict in self.rule_list:
            self.df.loc[:, 'vals'] = replace_values(self.df['vals'], rule_dict)
        if errors is not None:
            if len(errors) != len(values):
                raise ValueError(
                    "Mismatch between len of values and errors."
                    )
            else:
                self.df.loc[:, 'errs'] = errors
        if t_deltas is not None:
            if len(t_deltas) != len(values):
                raise ValueError(
                    "Mismatch between len of values and t_deltas."
                    )
            else:
                self.df.loc[:, 't_deltas'] = t_deltas
        # drop invalid vals (NaNs)
        # self.df_dropna = self.df.dropna(subset=['vals'], inplace=False)
        self.df_dropna = self.df[~ self.df['vals'].isnull()]
        self.vals = self.df_dropna['vals']
        self.N = len(self.df_dropna)
        self.weight_fn = weight_fn

    def SampleMean(self):
        # Sample mean of the experimental distribution, denoted \bar{x}, is
        # defined as \frac{1}{N}\sum{x_i}. The mean, \mu, of the *parent*
        # population, is the limit as N->inf of \bar{x}.  In general,
        # (parent parameter) = lim(N->inf)(experimental parameter)
        self.sample_mean = (1.0/self.N)*self.vals.sum()
        # Sample variance of the experimental distribution, denoted s^2, is
        # defined as \frac{1}{N-1}\sum{(x_i - \bar{x})^2}. The factor N-1
        # (instead of N) is due to the fact that \bar{x} has been determined
        # from the data and not independently.
        self.sample_variance = (
            (1.0/(self.N - 1.0)) * ((self.vals - self.sample_mean)**2).sum()
            )
        # The sample standard deviation s of the experimental distribution
        # is the square root of the sample variance. Note that the standard
        # deviation of the *parent* population, \sigma, is the sqrt of the
        # average of the squares of the deviations from the parent mean, \mu,
        # \simga^2 == \lim_{N-\inf}[\frac{1}{N}\sum{(x_i - \mu)^2}] =
        # \lim_{N-\inf}[\frac{1}{N}\sum{(x_i)^2}] - \mu^2.
        # In principle, s is a decent estimate of \simga for large enough N,
        # and s should be consistent with the standard deviation estimated
        # from considerations of the experimental equipment and measurement
        # conditions.
        self.sample_std = np.sqrt(self.sample_variance)
        # From error propogation considerations, we can estimate the
        # standard deviation in the mean (aka the standard error) \simga_\mu.
        # Assuming that the uncertainties in each measurement, \simga_i,
        # are all equal (\sigma_i = \sigma), then we have
        # \sigma_\mu = \frac{\simga}{\sqrt(N)} \simeq \frac{s}{\sqrt(N)}
        self.std_of_mean = self.sample_std / np.sqrt(self.N)
        # Note that the standard deviation of the *data*
        # does not decrease with repeated observations, it just becomes
        # better determined. However, the standard devaiton of the *mean*
        # decreases as the sqrt of N, indicating an improvement in our ability
        # to estimate \mu as we add more data.

    def WeightedMean(self):
        # Calculate a series of weights as defined by the weight function
        wi = (self.weight_fn(self.df_dropna))  # .astype('float')
        # Store these weights in the object dataframe for reference
        self.df.loc[:, 'wi'] = wi
        # Weighted mean is \frac{\sum{w_i*x_i}}{\sum{w_i}}
        self.weighted_sample_mean = (
            wi * self.df_dropna['vals']).sum() / (wi.sum())
        # Estimate of the sample standard deviation s of the experimental
        # distribution
        self.weighted_sample_std = np.sqrt(
            (self.N/(self.N - 1.0)) * (
                wi * (self.df_dropna['vals'] -
                      self.weighted_sample_mean)**2).sum() * 1.0/(wi.sum()))
        # Standard error, aka the standard deviation in the weighted mean
        self.std_of_weighted_mean = self.weighted_sample_std / np.sqrt(self.N)


def no_weight(qdf):
    '''Placeholder weight function, returning a series of ones of length
    equal to the length of qdf['vals']. The weighted mean should equal the
    sample mean when this function is used.'''
    return qdf['vals']*0 + 1.0


def weight_by_uncertainty(qdf):
    '''In circumstances where some data points have been measured with worse
    precision than others, we can express this quantitatively by assuming the
    measurements were sampling parent distributions with the same mean \mu
    but with different standard deviations \sigma_i.  In the calculation of
    the mean, we then weight each data point x_i inversely by its own
    variance \sigma_i^2
    '''
    return qdf['errs']**-2


def forgetting_factor(qdf, const=1, power=-1):
    '''In circumstances where the underlying distribution is changing with
    time, it may be desirable to weight older datapoints less than more
    recent ones. '''
    t_deltas = qdf['t_deltas']
    if (t_deltas <= 0).any():  # If any are <= 0, raise warning
        print("Warning: t_deltas contains non-positive values.")
    return (const*t_deltas)**power


def example_1():
    '''An example showing the effects of both removing outliers and weighting
    by time.
    '''
    vals =     [95.00, 98.30, 92.10, 22151.10, 77.80, 102.3, np.nan, 50.5, 25.0, -13.5, 20.0, 20.1]
    t_deltas = [1.5,   2.0,   2.2,   3.4,      20.8,  26.1,  30.23,  84.9, 90.2, 120.2, 150.0, 201]
    replace_rules = [{'operator': '>=', 'comparison_val': 150, 'replace_val': np.nan},
                     {'operator': '<=', 'comparison_val': -50, 'replace_val': np.nan},
                     {'operator': '>', 'comparison_val': 100, 'replace_val': 100},
                     {'operator': '<', 'comparison_val': 0, 'replace_val': 0}
                     ]
    qs = qSeries(vals,
                 t_deltas=t_deltas,
                 rule_list=replace_rules,
                 weight_fn=forgetting_factor)
    qs.SampleMean()
    qs.WeightedMean()
    print(qs.df)
    print("\nTotal inputs: {} ({} valid)".format(qs.N_raw, qs.N))
    print("Sample Mean: {0:.3f} +/- {1:.3f}".format(qs.sample_mean,
                                                    qs.std_of_mean))
    print("Sample STD: {0:.3f}".format(qs.sample_std))
    print("Weighted Mean: {0:.3f} +/- {1:.3f}".format(qs.weighted_sample_mean,
                                                      qs.std_of_weighted_mean))
    print("Weighted Sample STD: {0:.3f}".format(qs.weighted_sample_std))
    print("\nOn raw data, mean is:  {0:.3f}".format(qs.df.vals_raw.mean()))
    print("             std is:   {0:.3f}".format(qs.df.vals_raw.std()))
    print("             median is {0:.3f}".format(qs.df.vals_raw.median()))


def example_2():
    '''An example showing the effects of weighting by measurement error
    '''
    vals = [5.5, 5.6, 5.8, 5.4, 5.5, 2.4, 3.5, 3.4, 6.2, 5.1]
    errs = [0.2, 0.1, 0.1, 0.2, 0.3, 2.0, 2.0, 2.0, 2.0, 0.2]
    replace_rules = []
    qs = qSeries(vals,
                 errors=errs,
                 rule_list=replace_rules,
                 weight_fn=weight_by_uncertainty)
    qs.SampleMean()
    qs.WeightedMean()
    print(qs.df)
    print("\nTotal inputs: {} ({} valid)".format(qs.N_raw, qs.N))
    print("Sample Mean: {0:.3f} +/- {1:.3f}".format(qs.sample_mean,
                                                    qs.std_of_mean))
    print("Sample STD: {0:.3f}".format(qs.sample_std))
    print("Weighted Mean: {0:.3f} +/- {1:.3f}".format(qs.weighted_sample_mean,
                                                      qs.std_of_weighted_mean))
    print("Weighted Sample STD: {0:.3f}".format(qs.weighted_sample_std))
    print("\nOn raw data, mean is:  {0:.3f}".format(qs.df.vals_raw.mean()))
    print("             std is:   {0:.3f}".format(qs.df.vals_raw.std()))
    print("             median is {0:.3f}".format(qs.df.vals_raw.median()))


def example_3():
    '''Sanity check that the weighted mean equals the sample mean when
    "no_weight" is used as the weight function.
    '''
    vals = [5.5, 5.6, 5.8, 5.4, 5.5, 2.4, 3.5, 3.4, 6.2, 5.1]
    errs = [0.2, 0.1, 0.1, 0.2, 0.3, 2.0, 2.0, 2.0, 2.0, 0.2]
    replace_rules = []
    qs = qSeries(vals,
                 errors=errs,
                 rule_list=replace_rules,
                 weight_fn=no_weight)
    qs.SampleMean()
    qs.WeightedMean()
    print(qs.df)
    print("\nTotal inputs: {} ({} valid)".format(qs.N_raw, qs.N))
    print("Sample Mean: {0:.3f} +/- {1:.3f}".format(qs.sample_mean,
                                                    qs.std_of_mean))
    print("Sample STD: {0:.3f}".format(qs.sample_std))
    print("Weighted Mean: {0:.3f} +/- {1:.3f}".format(qs.weighted_sample_mean,
                                                      qs.std_of_weighted_mean))
    print("Weighted Sample STD: {0:.3f}".format(qs.weighted_sample_std))
    print("\nOn raw data, mean is:  {0:.3f}".format(qs.df.vals_raw.mean()))
    print("             std is:   {0:.3f}".format(qs.df.vals_raw.std()))
    print("             median is {0:.3f}".format(qs.df.vals_raw.median()))
