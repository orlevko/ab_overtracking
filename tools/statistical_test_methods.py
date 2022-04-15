from scipy.stats import chi2_contingency
import scipy.stats

__all__ = [
    't_test',
    'g_test',
]


def t_test(mean_base, stddev_base, visitors_base, mean_variant, stddev_variant, visitors_variant, alternative=None, equal_var=False):
    # TODO: check degrees of freedom for the standard deviation calculation
    # TODO: instead of calling ttest_ind_from_stats, we can adapt this function to call t_test_one_sample
    t_value, p_value = scipy.stats.ttest_ind_from_stats(mean_variant,
                                                        stddev_variant,
                                                        visitors_variant,
                                                        mean_base,
                                                        stddev_base,
                                                        visitors_base,
                                                        equal_var=equal_var  # False for Welch's t-test for unequal variances
                                                        )
    if alternative == 'greater':
        # Ha: mean_var - mean_base + delta >= 0
        if t_value > 0:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value/2
    elif alternative == 'less':
        # Ha: mean_var - mean_base + delta <= 0
        if t_value < 0:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value / 2

    return t_value, p_value


def g_test(counts_base, visitors_base, counts_variant, visitors_variant):
    # The g-test is only used for a two-sided test on binary metrics
    # TODO: return test statistic?
    try:
        chi2, p_value, _, _ = chi2_contingency([[counts_base, counts_variant],
                                                [visitors_base - counts_base, visitors_variant - counts_variant]],
                                               correction=False,  # This disables Yate's continuity correction
                                               lambda_='log-likelihood'  # Use G-test, instead of the default Chi square test
                                               )
    except ValueError:
        chi2, p_value = None, None
    return chi2, p_value
