from __future__ import division, print_function
from scipy.stats import norm, chi2, t
import numpy as np

def confidence_interval_asymptotic(confidence, mean, stdev, obs):
    '''
    Computes normal confidence interval around the mean
    ----
    Input
    ----
    confidence - confidence level for interval, 0 < x < 1
    mean - sample mean
    stdev - sample standard deviation
    obs - number of observations
    ----
    Output
    ----
    mean, lower bound, upper bound
    '''
    if (stdev < 0 or confidence <= 0 or
        confidence >=1 or obs <= 0):
        return np.nan, np.nan, np.nan
    zscore = norm.ppf(1 - (1-confidence)/2)
    confidence_interval = zscore * stdev / sqrt(obs)
    return mean, mean - confidence_interval, mean + confidence_interval

def confidence_interval_binomial(confidence, successes, obs):
    '''
    Computes assymptotic confidence inter`val for binomial distribution
    ----
    Input
    ----
    confidence - confidence level for interval, 0 < x < 1
    mean - number of successes
    obs - number of observations
    ----
    Output
    ----
    mean, lower bound, upper bound
    '''
    if (not obs) or (obs < 0) or (not successes) or (sucesses > obs):
        return np.nan, np.nan, np.nan 

    zscore = norm.ppf(1 - (1-confidence)/2)
    p = successes / np.float(total);
    confidence_interval = zscore * sqrt( p * ( 1 - p ) / total )
    
    return p, p-confidence_interval, p + confidence_interval

def improv_interval_binomial(confidence, successes_base, successes_var, obs_base, obs_var):
    '''
    Computes assymptotic confidence interval for ratio of means of binomial distributions. Uses confidence_interval_ratio,
    ----
    Input
    ----
    confidence - confidence level for interval, 0 < x < 1
    successes_base - number of successes in base
    successes_var - number of successes in variant
    obs_base - number of observations in base
    obs_variant - number of observations in variant
    ----
    Output
    ----
    impact, lower bound, upper bound
    '''
    if (not obs_base
        or not obs_var
        or (successes_base > obs_base)
        or (successes_var > obs_var) ):
        return np.nan, np.nan, np.nan

    avg_base = successes_base/obs_base
    avg_var = successes_var/obs_var
    stdev_base = np.sqrt((avg_base-avg_base**2))
    stdev_var = np.sqrt((avg_var-avg_var**2))

    return confidence_interval_ratio(confidence, avg_base, avg_var,
                                     stdev_base, stdev_var, obs_base, obs_var)

def confidence_interval_ratio(confidence, avg_base, avg_var, stdev_base, stdev_var,
                              obs_base, obs_var):
    '''
    Computes assymptotic confidence interval for ratio of means. Uses sample average, instead of sample mean average as in Math.pm
    ----
    Input
    ----
    confidence - confidence level for interval, 0 < x < 1
    avg_base - sample mean in base
    avg_var - sample mean variant
    stdev_base - sample standard deviation in base
    stdev_var - standard deviation in variant
    obs_base - number of observations in base
    obs_variant - number of observations in variant
    ----
    Output
    ----
    impact, lower bound, upper bound
    '''
    if  (not avg_base or
        np.isnan(avg_var)):
        return np.nan, np.nan, np.nan

    zscore = norm.ppf(1 - (1-confidence)/2)
    estimate = (avg_var - avg_base) / abs(avg_base)

    if (np.isnan(stdev_base) or
        np.isnan(stdev_var) or
        not obs_base or
        not obs_var):
        return estimate, np.nan, np.nan

    A = avg_base*avg_var
    B = avg_base**2 - (zscore**2 * stdev_base**2/obs_base)
    C = avg_var**2  - (zscore**2 * stdev_var**2/obs_var)
    sqrt = A**2 - (B*C)

    if (sqrt <= 0 or B<=0):
        return estimate, np.nan, np.nan

    range = np.sqrt(sqrt)/B
    fieller = [A/B - range - 1, A/B + range - 1]

    if (avg_base < 0):
        fieller = [-x for x in fieller]
    return estimate, fieller[0], fieller[1]

def g_test(successes_base, successes_var, obs_base, obs_var):
    '''
    G-test for means of binomial distribution
    ----
    Input
    ----
    successes_base - number of successes in base
    successes_var - number of successes in variant
    obs_base - number of observations in base
    obs_variant - number of observations in variant
    ----   
    Output
    ----
    G-test statistic (distributed as X^2(1) in H0)
    p-value
    '''
    # alternative: scipy.chi2_contingency() - implemented for completeness
    p = obs_base - successes_base
    q = successes_base
    r = obs_var - successes_var
    s = successes_var
    _sum = p + q + r + s

    if any([ x <=0 for x in [p, q, r, s] ]):
        return np.nan, np.nan

    P = (p + q)*(p + r) / _sum;
    Q = (p + q)*(q + s) / _sum;
    R = (r + s)*(p + r) / _sum;
    S = (r + s)*(q + s) / _sum;

    g_test = 2 * np.sum([
        p * np.log(p / P),
        q * np.log(q / Q),
        r * np.log(r / R),
        s * np.log(s / S)]
    )
    pvalue = 1 - chi2.cdf(g_test,1);
    return g_test, pvalue

def t_test(avg_base, avg_var, stdev_base, stdev_var,
                              obs_base, obs_var, two_tailed = True):
    ''''
    Welch's t-test for equality of means of normally distributed samples with unequal variance
    ----
    Input
    ----
    avg_base - sample mean in base
    avg_var - sample mean variant
    stdev_base - sample standard deviation in base
    stdev_var - standard deviation in variant
    obs_base - number of observations in base
    obs_variant - number of observations in variant
    two_tailed - two-sided (H1: v1 != v0) or one-sided (v1 > v0), default: True
    ----
    Output
    ----
    t-statistic
    p-value
    '''
    if (stdev_base <= 0) & (stdev_var <= 0):
        return np.nan, np.nan

    mean_variances = [stdev_base**2/obs_base,
                      stdev_var**2/obs_var]

    t_value = (avg_var - avg_base)/np.sqrt(sum(mean_variances))
    t_df = int( sum(mean_variances)**2 /
              ( ( mean_variances[0]**2 / (obs_base - 1) )
               + ( mean_variances[1]**2 / (obs_var - 1) ) ) )
              
    if t_df <= 0:
        return np.nan, np.nan

    if two_tailed:
        t_prob = t.cdf(abs(t_value), t_df)
        return t_value, 2*(1-t_prob)
    else:
        t_prob = t.cdf(t_value, t_df)
        return t_value, 1-t_prob
        
def stddev_metric_per_visitor_from_metric_per_booker(visitors, bookers, stddev_metric_per_booker, total_metric):
    """
    [adapted from perl ET function stddev_metric_per_visitor_from_metric_per_booker in https://gitlab.booking.com/core/main/blob/trunk/lib/Bookings/Experiment/Tool/Math.pm#L1252]
    We can compute the stddev of bookings/visitor from the known stddev of bookings/booker.
    Namely, all non-bookers have exactly 0 bookings/visitor.
    We know that the bookers form a population with population mean total_metric / $bookers
    and population variance stddev_metric_per_booker**2.
    By definition of the population variance, this means that
       [1] sum(metric**2)/bookers = stddev_metric_per_booker**2 + (total_metric / bookers)**2
    The bookers together with the non bookers form a population with population mean total_metric / visitors
    and population variance given by definition by:
       [2] variance_per_visitor = sum(metric**2)/visitors - (total_metric / visitors)**2
    If we substitute the expression for sum(metric**2) from [1] into [2], we obtain
       [3] variance_per_visitor =
              (bookers/visitors) * ( stddev_metric_per_booker**2 + (total_metric / bookers)**2 )
                                 - (total_metric / visitors)**2
    Expanding the parentheses gives the formula below.

    ----
    Input
    ----
    visitors - number of visitors
    bookers - number of bookers
    stddev_metric_per_booker - standard deviation of metric per booker
    total_metric - sum of the metric (then used to get average per booker or per visitor)
    ----
    Output
    ----
    standard deviation of metric per visitor
    """     
    if ((bookers==0) or (visitors==0) or (visitors < 2) or (stddev_metric_per_booker==0)):
        return None

    non_bookers = visitors - bookers
    variance = (
          (bookers / visitors)    * stddev_metric_per_booker**2
        + (non_bookers / bookers) * (total_metric / visitors)**2
    )
    if (variance<=0):
        return None
    else:
        return math.sqrt(variance)
