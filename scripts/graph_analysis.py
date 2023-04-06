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


# Generate graphs and extract metrics

thresholds = np.arange(55,96, 1)

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
zscores = pd.read_csv(f"{data_dir}/Data_zscores.csv", index_col="SubjID")
demo_columns = demographics.columns
brain_regions = zscores.columns

def joint_variation(v):
    warnings.filterwarnings('ignore')
    if ~isinstance(v, np.ndarray):
        v = np.asanyarray(v)
    JV = np.array([ 1 / np.exp((v - i) ** 2) for i in v])
    np.fill_diagonal(JV, 0)
    return JV

def get_metrics(M, thr, nodes):
    M[M < np.percentile(M, thr)] = 0
    M[M > 0] = 1
    
    B = nx.from_numpy_matrix(M)
    B = nx.relabel_nodes(B, dict(zip(B, nodes)))
    B = nx.algorithms.full_diagnostics(B, swi=False, swi_niter=100, swi_nrand=10, swi_seed=None, n_jobs=nj, prefer=None)
    
    attributes = B.nodes[nodes[0]].keys()
    metric_dict = {metric: nx.get_node_attributes(B, metric) for metric in attributes}
    metric_dict.update({metric: {"global": value} for metric, value in B.metrics.items()})
    return metric_dict


results = {}
logging.debug(f"\n\nCOMPUTED METRICS FOR K:")
for thr in thresholds:
    r = Parallel(n_jobs=nj)(delayed(get_metrics)(joint_variation(v), thr, brain_regions) for _, v in zscores.iterrows())
    results.update({thr: dict(zip(zscores.index, r))})
    logging.debug(f"\t{thr}")

output_df = pd.DataFrame([[subj, thr] for subj in results[thr].keys() for thr in results.keys()], columns=['SubjID', 'Density']).set_index('SubjID')

metrics = results[thresholds[0]][zscores.index[0]].keys()
for metric in metrics:
    columns = [(node, 
               metric, 
              [results[density][subj][metric][node] for subj, density in output_df['Density'].iteritems()]) 
              for node in results[thresholds[0]][zscores.index[0]][metric].keys()]
    
    for node, metric, values in columns:
        output_df[f"{node}_{metric}"] = np.float64(values)

        
output_df.to_csv(f"{output_dir}/Graph_metrics.csv")

logging.debug(f"GRAPH METRICS COMPUTED\n{log_func(locals())}\n\n")
