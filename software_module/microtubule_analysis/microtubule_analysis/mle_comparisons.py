"""Compares maximum likelihood estimators for the data in diffent ways."""
from .calculate_mles import * 
import pandas as pd
import numpy as np
import bokeh_catplot

def tidy_up_data(bad_df):
    """Tidies up the multiple-concentration tubule catastrophe data."""
    df = bad_df.copy()
    new_cols = np.empty(len(df.columns))

    for i, val in enumerate(df.columns):
        end_of_num = val.rfind(' uM')
        new_cols[i] = val[ : end_of_num]

    df.columns = new_cols
    df.columns.name = 'Tubulin Concentration (uM)'
    df = df.melt(value_name='Time to Catastrophe (s)')
    df = df[df['Time to Catastrophe (s)'].notna()]
    df = df.sort_values('Tubulin Concentration (uM)')
    df = df.reset_index(drop=True)

    return df

def plot_ecdf(tidy_data, cats, val, title, width=550, conf_int=False):
    """
    Plots an ECDF of tidy data.
    tidy_data: Set of tidy data.
    cats: Categories to plot
    val: The value to plot
    title: Title of plot
    width: width of plot
    conf_int: Whether or not to bootstrap a CI.

    """
    p = bokeh_catplot.ecdf(
        data = tidy_data,
        cats = cats,
        val = val,
        title = title,
        width = width,
        conf_int = conf_int,
    )
    return p

def get_data_from_concentration(df, concentration):
    """
    Gets a numpy array of the data given the concentration and a df.
    """
    
    data_df = df[df['Tubulin Concentration (uM)'] == concentration]
    
    return data_df['Time to Catastrophe (s)'].values

def get_aic_weight(log_likelihood_1, log_likelihood_2):
    """
    Returns the AIC weight of the first model.
    """
    aic_1 = 4 - 2 * log_likelihood_1
    aic_2 = 4 - 2 * log_likelihood_2

    aic_max = max(aic_1, aic_2)
    num = np.exp(-(aic_1 - aic_max)/2)
    denom = np.exp(-(aic_1 - aic_max)/2) + np.exp(-(aic_2 - aic_max)/2)

    return num / denom

def get_all_confidence_intervals_gamma(tidy_data):
    tubulin_concentrations = np.unique(tidy_data["Tubulin Concentration (uM)"])

    ci_dictionary = {
        'Concentration' : [],
        'Alpha': [],
        'Beta': [],
        'CI_alpha': [],
        'CI_beta': [],
    }
    for conc in tubulin_concentrations:
        ci_dictionary['Concentration'].append(conc)
        conc_data = get_data_from_concentration(tidy_data, conc)
        alpha, beta = mle_fun_gamma(conc_data)
        
        bs_reps = draw_bs_reps_mle(
            mle_fun_gamma, 
            conc_data, 
            size=1000, 
        )

        conf_int = np.percentile(bs_reps, [2.5, 97.5], axis=0)    
        
        ci_dictionary['Alpha'].append(alpha)
        ci_dictionary['Beta'].append(beta)

        ci_dictionary['CI_alpha'].append(conf_int[:, 0])
        ci_dictionary['CI_beta'].append(conf_int[:, 1])

    return pd.DataFrame(data = ci_dictionary)
        