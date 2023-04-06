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


# Correlation coefficients

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
gmetrics = pd.read_csv(f"{output_dir}/Graph_metrics.csv", index_col="SubjID")
metrics = gmetrics.columns.to_list()


Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 5)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+10]])

for Kmin, Kmax in K_ranges:
    AUC = gmetrics[(gmetrics.Density >= Kmin) & (gmetrics.Density <= Kmax)].groupby('SubjID').sum()
    AUC = AUC.merge(demographics, left_index=True, right_index=True)
    
    
    group_Rs = []
    for group in demographics.Dx3.unique():
        R = AUC[AUC.Dx3==group].corr(method='spearman')[demographics.columns]
        R.Dx3 = group
        group_Rs.append(R)
        
    R_table = pd.concat(group_Rs).sort_values('Dx3')
    R_table['Kmin'] = Kmin
    R_table['Kmax'] = Kmax
    
    R_table.to_csv(f"{output_dir}/Correlations_K{Kmin}-{Kmax}.csv")
        
        
logging.debug(f"CORRELATIONS COMPUTED\n{log_func(locals())}\n\n")
