# Import dependencies

from directories import data_dir, covariates, thickness, volume, output_dir, nj
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import warnings
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging
import argparse

logging.basicConfig(filename=f"{output_dir}/log.txt",
    format='%(asctime)s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S %Z')
def log_func(var_dict):
	new = {key:type(value) for key, value in var_dict.items()}
	return new
	
#-----------------------------------------------------------------------------------------------------------------


# Regression models

import statsmodels.formula.api as sm


demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
gmetrics = pd.read_csv(f"{output_dir}/Graph_metrics.csv", index_col="SubjID")[demographics.Dx3==2]
global_metrics = gmetrics.columns[gmetrics.columns.str.startswith('global')]

def modeldata(formula, data):
    model = sm.gls(formula, data=data)
    fitted = model.fit()
    
    endog = pd.Series('_'.join(model.endog_names.split('_')[1:]), index=['endog'])
    params = fitted.params.rename(index=dict(zip(fitted.params.index, 'param_' + fitted.params.index)))
    errors = fitted.bse.rename(index=dict(zip(fitted.params.index, 'bse_' + fitted.bse.index)))
    pvalues = fitted.pvalues.rename(index=dict(zip(fitted.pvalues.index, 'pvalues_' + fitted.bse.index)))
    rsquared = pd.Series(fitted.rsquared, index=['rsquared'])
    rsquared_adj = pd.Series(fitted.rsquared_adj, index=['rsquared_adj'])
    
    results = pd.concat([endog, params, errors, pvalues, rsquared, rsquared_adj])
    return results

Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 5)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+10]])

for Kmin, Kmax in K_ranges:
    AUC = gmetrics[(gmetrics.Density >= Kmin) & (gmetrics.Density <= Kmax)].groupby('SubjID').sum()
    AUC = AUC.merge(demographics, left_index=True, right_index=True)
    
    regression_results = Parallel(n_jobs=nj)(delayed
                                             (modeldata)
                                             (f"{metric} ~ Age * Bmi", data=AUC) 
                                             for metric in global_metrics)
    
    regression_table = pd.concat([res for res in regression_results], axis=1).T.set_index('endog')
    

    regression_table['Kmin'] = Kmin
    regression_table['Kmax'] = Kmax
    
    regression_table.to_csv(f"{output_dir}/Regressions_K{Kmin}-{Kmax}.csv")
        
        
logging.debug(f"REGRESSION MODELS COMPUTED\n{log_func(locals())}\n\n\n\n")
#-----------------------------------------------------------------------------------------------------------------
