import numpy as np
import scipy.stats

from app.constants import DEFAULT_ALPHA

__all__ = [
    'confidence_interval',
    'confidence_interval_margin_norm',
    'confidence_interval_to_margin',
    'confidence_interval_absolute',
    'confidence_interval_ratio',
]


def confidence_interval_margin_norm(stddev, visitors, alpha=DEFAULT_ALPHA):
    if stddev is None or visitors is None or visitors == 0:
        return None

    zscore = scipy.stats.norm.ppf(1 - alpha/2)
    standard_error = stddev / np.sqrt(visitors)
    return zscore * standard_error


def confidence_interval_to_margin(point_estimate, lower_bound, upper_bound):
    if point_estimate is None or np.isnan(point_estimate) or lower_bound is None or upper_bound is None or np.isnan(lower_bound) or np.isnan(upper_bound):
        return None
    
    margin_of_error = max(np.abs(point_estimate - lower_bound), np.abs(upper_bound - point_estimate))
    return float(margin_of_error)


def confidence_interval(point_estimate, stddev, visitors, alpha=DEFAULT_ALPHA):
    margin = confidence_interval_margin_norm(stddev, visitors, alpha)

    lower_bound = None
    upper_bound = None

    if margin is not None:
        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

    return point_estimate, lower_bound, upper_bound, margin


def confidence_interval_absolute(avg_base, stdev_base, obs_base, avg_var, stdev_var, obs_var, alpha=DEFAULT_ALPHA):
    # TODO: Replace this with proper confidence intervals
    if avg_var is None or avg_base is None:
        return None, None, None
    
    point_estimate = avg_var - avg_base
    
    if stdev_base is None or stdev_var is None or np.isnan(stdev_base) or np.isnan(stdev_var):
        return point_estimate, None, None
    
    stddev = np.sqrt(stdev_base**2/obs_base + stdev_var**2/obs_var)
    zscore = scipy.stats.norm.ppf(1 - alpha/2)
    margin = zscore * stddev
    lower_bound = point_estimate - margin
    upper_bound = point_estimate + margin

    return point_estimate, lower_bound, upper_bound


def confidence_interval_ratio(avg_base, stdev_base, obs_base, avg_var, stdev_var, obs_var, alpha=DEFAULT_ALPHA):
    '''
    This should be used only to ensure parity with the current ET implementation. Down the line, better intervals should be used.

    Adapted from et_math.py: https://gitlab.booking.com/datascience/pydat/blob/stats/booking/stats/et_math.py#L80
    Computes asymptotic confidence interval for ratio of means. Uses sample average, instead of sample mean average as in Math.pm
    ----
    Input
    ----
    avg_base - sample mean in base
    stdev_base - sample standard deviation in base
    obs_base - number of observations in base
    avg_var - sample mean variant
    stdev_var - standard deviation in variant
    obs_variant - number of observations in variant
    alpha - false positive rate, 0 < x < 1
    ----
    Output
    ----
    impact, lower bound, upper bound
    '''
    if (not avg_base or
            avg_var is None or np.isnan(avg_var)):
        return None, None, None

    estimate = (avg_var - avg_base) / abs(avg_base)

    if (stdev_base is None or stdev_var is None or 
        np.isnan(stdev_base) or np.isnan(stdev_var) or
        not obs_base or not obs_var):
        return estimate, None, None

    zscore = scipy.stats.norm.ppf(1 - alpha/2)
    A = avg_base*avg_var
    B = avg_base**2 - (zscore**2 * stdev_base**2/obs_base)
    C = avg_var**2 - (zscore**2 * stdev_var**2/obs_var)
    sqrt = A**2 - (B*C)

    if (sqrt <= 0 or B <= 0):
        return estimate, np.nan, np.nan

    range = np.sqrt(sqrt)/B
    fieller = [A/B - range - 1, A/B + range - 1]

    if (avg_base < 0):
        fieller = [-x for x in fieller]
    return estimate, fieller[0], fieller[1]
