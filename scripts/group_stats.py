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


# Group stats

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
gmetrics = pd.read_csv(f"{output_dir}/Graph_metrics.csv", index_col="SubjID")
metrics = gmetrics.columns.to_list()
stat_list = [np.nanmean, np.nanstd, np.nanmedian, stats.iqr, 'count']
groups = ['Dx', 'Dx3', 'Subtype', 'Minor', 'Durill3']



stat_list = ['count', np.nanvar, np.nanmean, np.nanstd, np.nanmedian, stats.iqr]
contrasts = [['Dx', 'Dx3'],
             ['Dx', 'Dx3','Minor'],  
             ['Dx', 'Dx3', 'Subtype'],
             ['Dx', 'Dx3', 'Durill3']]



Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 5)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+10]])

for Kmin, Kmax in K_ranges:
    AUC = gmetrics[(gmetrics.Density >= Kmin) & (gmetrics.Density <= Kmax)].groupby('SubjID').sum()
    AUC = AUC.merge(demographics, left_index=True, right_index=True)

    group_stats = [AUC.groupby(groups).agg(stat_list).stack().reset_index().rename(columns={f'level_{len(groups)}':'Stat'})
                   for i, groups in enumerate(contrasts)]
    
    for i, data in enumerate(group_stats):
        data.sort_values(contrasts[i] + ['Stat'])
        data['Contrast'] = i

    aggregation_table = pd.concat(group_stats).set_index(['Contrast', 'Dx', 'Dx3', 'Subtype', 'Minor', 'Durill3']).reset_index()
    aggregation_table['Kmin'] = Kmin
    aggregation_table['Kmax'] = Kmax
    
    aggregation_table.to_csv(f"{output_dir}/Group_stats_K{Kmin}-{Kmax}.csv")
        
        
logging.debug(f"GROUP STATS COMPUTED\n{log_func(locals())}\n\n")
