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


# Correct for covariates

def covar_correct(X, Y, data, n_jobs=1):
    
    def get_resid(X, y, data, n_jobs=n_jobs):
        r = data[y] - LinearRegression(n_jobs=n_jobs).fit(data[X], data[y]).predict(data[X])
        return r
    
    R = Parallel(n_jobs=n_jobs)(delayed(get_resid)(X, y, data, n_jobs=n_jobs) for y in Y)
    return np.asanyarray(R).T

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
thickness_volume = pd.read_csv(f"{data_dir}/CT_Volume.csv", index_col="SubjID")
brain_regions = thickness_volume.columns                               
                               
data = demographics[['Age', 'Hand']].merge(thickness_volume, left_index=True, right_index=True)
                               
X = ['Age', 'Hand']
Y = brain_regions

residuals = thickness_volume.copy()
residuals[brain_regions] = covar_correct(X, Y, data, n_jobs=nj)


residuals.to_csv(f"{data_dir}/Data_residuals.csv")

logging.debug(f"DATA CORRECTED FOR CONFOUNDS\n{log_func(locals())}\n\n")
