from statsmodels.stats.power import tt_ind_solve_power

from app.constants import DEFAULT_ALPHA, DEFAULT_BETA

__all__ = [
    'calculate_sample_size',
    'calculate_effect_size',
    'calculate_power',
]

# Returns the total sample size
def calculate_sample_size(
    standard_deviation,
    effect_size,
    alpha = DEFAULT_ALPHA,
    beta = DEFAULT_BETA,
    ratio = 1,
    n_test = 1,
    # We accept two-sided, larger, smaller because in the future we can add superiority tests.
    alternative = 'two-sided',
):  
    base_sample = tt_ind_solve_power(
        effect_size = effect_size / standard_deviation,
        alpha = alpha,
        power = (1 - beta),
        ratio = ratio,
        alternative = alternative,
    )

    return (1 + (ratio * n_test)) * base_sample


# Returns the effect_size for comparative test and the acceptable cost for one-sided tests
def calculate_effect_size(
    sample_size,
    standard_deviation,
    #TODO: expected effect shouldn't change signs in case of NI 
    #TODO: expected effect and MRE should not be confused. Stick to standard statistics naming conventions: delta1 = parameters under H1, delta0 = parameters under H0 and delta = delta1 - delta0
    expected_effect = 0, # if alternative != 'larger' than positive expected_effect means a degredation
    alpha = DEFAULT_ALPHA,
    beta = DEFAULT_BETA,
    ratio = 1,
    n_test = 1,
    # We accept two-sided, larger, smaller because in the future we can add superiority tests.
    alternative = 'two-sided',
):

    if alternative == 'two-sided' and expected_effect != 0:
        raise ValueError('two-sided test does not accept an expected_effect')
    
    base_sample_size = sample_size / (1 + (ratio * n_test))
    standarized_effect_size = tt_ind_solve_power(
        nobs1 = base_sample_size,
        alpha = alpha,
        power = (1 - beta),
        ratio = ratio,
        alternative = alternative
    )
    # We assume that the expected effect for NI is constant and we want to return the acceptable cost
    return (standard_deviation * standarized_effect_size) + expected_effect


def calculate_power(
    standard_deviation,
    sample_size,
    effect_size,
    alpha = DEFAULT_ALPHA,
    ratio = 1,
    n_test = 1,
    # We accept two-sided, larger, smaller because in the future we can add superiority tests.
    alternative = 'two-sided',
):  
    base_sample_size = sample_size / (1 + (ratio * n_test))
    power = tt_ind_solve_power(
        nobs1 = base_sample_size,
        effect_size = effect_size / standard_deviation,
        alpha = alpha,
        ratio = ratio,
        alternative = alternative
    )
