"""Generates Confidence Intervals for Microtubule Catastrophe in various
ways."""

import numpy as np
import pandas as pd
import scipy.stats
import bokeh.plotting as bplot
import bokeh.io

def generate_sample_reps(data, function, samples=1):
    """
    Generates the bootstrap replicates.
    Parameters:
    data: The input data
    function: The function being used to generate the sample replicates
    samples: The number of samples.
    """
    return [
            function(np.random.choice(data, size=len(data))) 
            for i in range(samples)
    ]

def generate_bs_confidence_interval(data, function, n_samples=1000):
    """
    Generates the confidence interval for a sample using bootstrapping.
    Parameters:
    data: The data we are using
    function: The function we are using for the confidence interval.
    n_samples: The number of samples we are bootstrapping
    """
    sample = generate_sample_reps(data, function, samples=n_samples)
    return np.quantile(sample, [0.025, 0.975])

def find_p_value_permutation(data1, data2, n_samples = 1000):
    """
    Finds the bootstrapped p-value using permutaion for data1 and data2.
    Parameters:
    data1: The first dataset.
    data2: The second dataset.
    n_samples: The number of times to permute.
    """

    test_statistic = np.abs(np.mean(data1) - np.mean(data2))

    concatenated = np.concatenate([data1, data2])

    number_of_times_test_is_greater_than_null_stat = 0

    for i in range(n_samples):
        # Generating the null statistic from the null distribution
        permuted = np.random.permutation(concatenated)
        null_sample_1 = permuted[:len(data1)]
        null_sample_2 = permuted[len(data1):]
        null_statistic = np.mean(null_sample_1) - np.mean(null_sample_2)
        
        if test_statistic >= np.abs(null_statistic):
            number_of_times_test_is_greater_than_null_stat += 1

    p = number_of_times_test_is_greater_than_null_stat / n_samples
    return p

def ci_using_normal(data):
    """
    Gets a confidence interval using a normal distribution on the data.
    Parameters:
    data: The data we are getting the confidence interval for.
    """
    n = len(data)

    mu = np.mean(data)
    sigma = np.sqrt(1/n)*np.std(data)

    lower = scipy.stats.norm.ppf(
        0.025, 
        loc=mu, 
        scale=sigma
    )
    upper = scipy.stats.norm.ppf(
        0.975, 
        loc=mu, 
        scale=sigma
    )

    return (lower, upper)

def ecdf(data):
    """
    Computes the ecdf of data.
    Parameters:
    data: The data whose ECDF we are computing.
    """
    x = np.arange(int(min(data)), int(max(data)))
    sorted_data = np.sort(data)
    ecdfs = []
    n = len(data)
    for val in x:
        ecdfs.append(len([d for d in data if d <= val])  / n)
    return ecdfs

def compute_dkw_bounds(ecdf, n, alpha=0.05):
    """
    Takes the ECDF and returns the upper and lower DKW bound.
    Parameters:
    ECDF: the ECDF we have calculated.
    n: The length of the data.
    alpha: The level of significance.
    """
    epsilon = np.sqrt((1/(2*n))*np.log(2/alpha))
    return max(ecdf - epsilon, 0), min(ecdf + epsilon, 1)

def plot_ecdf_dkw(data, data_title='', alpha=0.05, p=None, color='orange'):
    """
    Plots an ECDF with DKW bounds. 
    Parameters:
    data: The data we're plotting.
    alpha: The alpha value we are going to use.
    """
    f_hat = ecdf(data)
    n = len(data)
    l = [compute_dkw_bounds(x, n)[0] for x in f_hat]
    u = [compute_dkw_bounds(x, n)[1] for x in f_hat]

    ecdf_bound = pd.DataFrame({
        'ecdf': f_hat, 
        'lower': l, 
        'upper': u,
        'x': np.arange(int(min(data)), int(max(data)))
    })
    if p is None:
        p = bplot.figure(
            width=400,
            height=300,
            x_axis_label='Time to catastrophe',
            y_axis_label='ECDF',
            title = 'ECDFs and Confidence Intervals'
        )
    p.circle(
        source=ecdf_bound,
        x='x',
        y='ecdf',
        color=color,
        legend=data_title
    )
    p.circle(
        source=ecdf_bound,
        x='x',
        y='lower',
        color=color,
    )
    p.circle(
        source=ecdf_bound,
        x='x',
        y='upper',
        color=color
    )

    p.legend.location = 'bottom_right'

    return p

def print_confidence_interval(ci_low, ci_high, title=''):
    print(title, 'confidence interval (95%) is: [{0:.5f}, {1:.5f}]'.format(
        ci_low,
        ci_high
    ))