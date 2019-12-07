"""Simulates a 2 Poisson Process Distribution."""

import numpy as np
import pandas as pd
import bokeh_catplot
import bokeh.io

rg = np.random.default_rng()

def generate_samples_2pp(beta1, beta2, n_samples=1):
    """
    Generates n_samples samples according to the 2 Poisson Process Distribution.
    Parameters:
    beta1: Rate of the first event.
    beta2: Rate of the second event.
    n_samples: The number of samples to generate.
    """
    times = []
    for i in range(n_samples):
        time = rg.exponential(1 / beta1) + rg.exponential(1 / beta2)
        times.append(time)
    return times

def generate_samples_gamma(a, b, n_samples=1):
    """
    Draws samples from a gamma distribution
    
    Parameters:
    a: The alpha term.
    b: The beta term (rate).
    size: The number of samples to draw
    """
    times = np.empty(n_samples)
    for i in range(n_samples):
        times[i] = rg.gamma(a, scale=1/b)
    return times

def create_generated_ecdfs(beta_1, beta_2, n_samples=150):
    """
    Creates a plot containing all generated ECDFs from the 2 Poisson Process
    Distribution. It holds beta 1 constant.
    Parameters:
    beta1: An integer representing beta 1.
    beta2: A list of integers representing beta 2.
    n_samples: The number of samples to generate each time.
    """

    
    times = {'times': [], 'beta 2': []}

    for val in beta_2:
        sampled_times = generate_samples_2pp(beta_1, val, n_samples=n_samples)
        beta_2_array = [val] * n_samples
        
        times['times'] += sampled_times
        times['beta 2'] += beta_2_array
        
        
        
    df = pd.DataFrame(data=times)
    df.head()
        
        
    p = bokeh_catplot.ecdf(
        data=df,
        cats=['beta 2'],
        val='times',
        style='staircase',
        title='Generated ECDFs for Differing Values of β'
    )

    p.legend.location = 'bottom_right'

    return p

def cdf_points(t, beta_1, beta_2):
    '''
    Gets the points for a given analytical CDF.
    Parameters:
    t: An array of t-values, 
    beta_1: A beta 1 value 
    beta_2: A beta 2 value.
    It returns an array of the CDF values.
    '''
    ratio = (beta_1 * beta_2) / (beta_2 - beta_1)
    
    term1 = (1 / beta_1) * (1 - np.exp(-beta_1 * t))
    term2 = (1 / beta_2) * (1 - np.exp(-beta_2 * t))
    
    return ratio * (term1 - term2)

def theo_empirical_cdf(beta_1, beta_2, n_samples=150):
    """
    Overlays empirical and theoretical ECDFs for the 2 Poisson Process Distribution.
    Parameters:
    beta_1: The beta 1 value
    beta_2: The beta 2 value.
    n_samples: The number of samples to generate.
    """
    # Gets our analytical ECDF
    times = np.arange(0, 20, .1)
    prob_times = cdf_points(times, beta_1, beta_2)

    t = {'times': [], 'beta 2': []}

    sampled_times = generate_samples_2pp(beta_1, beta_2, n_samples=n_samples)
    beta_2_array = [beta_2] * n_samples
    
    t['times'] += sampled_times
    t['beta 2'] += beta_2_array
    

    df = pd.DataFrame(data=t)
    df['label'] = 'Empirical CDF'

    p = bokeh_catplot.ecdf(
        data=df,
        cats='label',
        val='times',
        style='staircase',
        show_legend=True,
        conf_int=True,
        title='Empirical and Theoretical CDFs'
    )
    p.circle(
        x=times,
        y=prob_times,
        size=2,
        color='orange',
        legend='Theoretical CDF',
    )

    p.yaxis.axis_label = 'CDF value'

    return p
