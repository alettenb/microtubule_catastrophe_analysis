"""Calculates the maximum likelihood estimators for gamma and 2 Poisson Process
distributions."""
import numpy as np
import tqdm

import scipy
import scipy.stats
import warnings

rg = np.random.default_rng()

def log_likelihood_function_gamma(params, n):
    """
    Gets the log likelihood for a gamma distribution.
    Params:
    params: alpha and beta, corresponding to a gamma dist. params
    n: The vector of data
    """
    alpha, beta = params
    
    if alpha <= 0 or beta <=0:
        return -np.inf
    
    return np.sum(scipy.stats.gamma.logpdf(n, a=alpha, scale=1/beta))

def mle_fun_gamma(n):
    """
    Finds the maximum likelihood parameters for a gamma distribution.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scipy.optimize.minimize(
            fun=lambda params, n: -log_likelihood_function_gamma(params, n),
            x0=np.array([3, .006]),
            args=(n,),
            method='Powell'
        )
    return result.x

def log_likelihood_function_2pp(params, n):
    """
    Gets the log likelihood function of two poisson distributions with 
    different rates happening after each other.
    params:
    params: the parameters representing the two rates.
    n: A numpy array of the data.
    """
    beta_1, beta_2 = params
    
    # To prevent identifiablility issues, we will make sure beta_2 > beta_1.
    if beta_1 <= 0 or beta_2 <=0 or beta_1 > beta_2:
        return -np.inf
    
    elif np.abs(beta_2 - beta_1) <= .00001:
        return np.sum(scipy.stats.gamma.logpdf(n, a=2, scale=1/beta_1))
    
    else:
        start_term = beta_1 * beta_2 / (beta_2 - beta_1)
        tot = 0
        for i, val in enumerate(n):
            tot += scipy.special.logsumexp(
                a=[-beta_1 * val, -beta_2 * val], 
                b=[start_term, -1 * start_term]
            )
        return tot

def mle_fun_2pp(n):
    """
    Gets our mle function for part b; two poisson distributions with different
    rates happening after each other.
    params:
    n: our vector of data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scipy.optimize.minimize(
            fun=lambda params, n: -log_likelihood_function_2pp(params, n),
            x0=np.array([100, 100]),
            args=(n,),
            method='Powell'
        )
    return result.x

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])