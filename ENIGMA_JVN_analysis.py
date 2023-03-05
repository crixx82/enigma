# Input data paths

data_dir = "/home/data"
covariates = f"{data_dir}/Covariates_simulation.csv"
thickness = f"{data_dir}/CorticalMeasuresENIGMA_ThickAvg.csv"
volume = f"{data_dir}/SubcorticalMeasuresENIGMA_VolAvg.csv"
output_dir = f"/home/output"

#-----------------------------------------------------------------------------------------------------------------


# Import dependencies

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import warnings
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(filename=f"{output_dir}/log", level=logging.DEBUG)
#-----------------------------------------------------------------------------------------------------------------


# Assemble dataframe for analyses and screen subjects

demographics = pd.read_csv(covariates, index_col="SubjID")
dashless_columns = pd.Series(demographics.columns.str.split('_')).apply(lambda x: ''.join(str.capitalize(s) for s in x))
demographics.rename(columns=dict(zip(demographics.columns, dashless_columns)), inplace=True)
demographics.loc[demographics.Dx3==0, 'Dx3'] = np.nan
demographics['Cohort'] = np.int32(demographics.Age<18.)

thickness_volume = pd.read_csv(thickness, index_col="SubjID", usecols=lambda x: x!='ICV').merge(pd.read_csv(volume, index_col="SubjID"), left_index=True, right_index=True)
dashless_columns = pd.Series(thickness_volume.columns.str.split('_')).apply(lambda x: ''.join(str.capitalize(s) for s in x))
thickness_volume.rename(columns=dict(zip(thickness_volume.columns, dashless_columns)), inplace=True)

macroscale_indices = ['Lthickness', 'Rthickness', 'Lsurfarea', 'Rsurfarea', 'Icv']
demographics = demographics.merge(thickness_volume[macroscale_indices], left_index=True, right_index=True)
thickness_volume = thickness_volume.drop(macroscale_indices, axis=1)


covar_regressors = ['Age', 'Hand']
to_keep = np.all(~demographics[covar_regressors].isna(), axis=1)
demographics = demographics.loc[to_keep, :]
thickness_volume = thickness_volume.loc[to_keep, :]


demographics.to_csv(f"{output_dir}/Demographics.csv")
thickness_volume.to_csv(f"{output_dir}/CT_Volume.csv")

logging.debug("\n\nDATAFRAMES GENERATED")

#-----------------------------------------------------------------------------------------------------------------


# Correct for covariates

def covar_correct(X, Y, data):
    
    def get_resid(X, y, data):
        r = data[y] - LinearRegression(n_jobs=1).fit(data[X], data[y]).predict(data[X])
        return r
    
    R = Parallel(n_jobs=-1)(delayed(get_resid)(X, y, data) for y in Y)
    return np.asanyarray(R).T

demographics = pd.read_csv(f"{output_dir}/Demographics.csv", index_col="SubjID")
thickness_volume = pd.read_csv(f"{output_dir}/CT_Volume.csv", index_col="SubjID")
brain_regions = thickness_volume.columns                               
                               
data = demographics[['Age', 'Hand']].merge(thickness_volume, left_index=True, right_index=True)
                               
X = ['Age', 'Hand']
Y = brain_regions

residuals = thickness_volume.copy()
residuals[brain_regions] = covar_correct(X, Y, data)


residuals.to_csv(f"{output_dir}/Data_residuals.csv")

logging.debug("\n\nDATA CORRECTED FOR CONFOUNDS")

#-----------------------------------------------------------------------------------------------------------------


# Normalize on healthy subjects

demographics = pd.read_csv(f"{output_dir}/Demographics.csv", index_col="SubjID")
residuals = pd.read_csv(f"{output_dir}/Data_residuals.csv", index_col="SubjID")
residuals = thickness_volume.merge(demographics['Dx'], left_index=True, right_index=True)

mu = residuals.groupby('Dx').mean()
sd = residuals.groupby('Dx').std()

zscores = ((residuals - mu.loc[0]) / sd.loc[0]).drop('Dx', axis=1)


zscores.to_csv(f"{output_dir}/Data_zscores.csv")

logging.debug("\n\nDATA ZSCORED")

#-----------------------------------------------------------------------------------------------------------------


# Generate graphs and extract metrics

thresholds = np.arange(95,101, 1)

demographics = pd.read_csv(f"{output_dir}/Demographics.csv", index_col="SubjID")
zscores = pd.read_csv(f"{output_dir}/Data_zscores.csv", index_col="SubjID")
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
    B = nx.algorithms.full_diagnostics(B)
    
    attributes = B.nodes[nodes[0]].keys()
    metric_dict = {metric: nx.get_node_attributes(B, metric) for metric in attributes}
    metric_dict.update({metric: {"global": value} for metric, value in B.metrics.items()})
    return metric_dict


results = {}
for thr in thresholds:
    r = Parallel(n_jobs=1)(delayed(get_metrics)(joint_variation(v), thr, brain_regions) for _, v in zscores.iterrows())
    results.update({thr: dict(zip(zscores.index, r))})

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

logging.debug("\n\nGRAPH METRICS EXTRACTED")

#-----------------------------------------------------------------------------------------------------------------


# Summary stats

demographics = pd.read_csv(f"{output_dir}/Demographics.csv", index_col="SubjID")
gmetrics = pd.read_csv(f"{output_dir}/Graph_metrics.csv", index_col="SubjID")
metrics = gmetrics.columns.to_list()
stat_list = [np.nanmean, np.nanstd, np.nanmedian, stats.iqr, 'count']
groups = ['Dx', 'Dx3', 'Subtype', 'Cohort']


stat_list = [ 'count', np.nanvar, np.nanmean, np.nanstd, np.nanmedian, stats.iqr]
contrasts = [['Dx'], 
             ['Dx', 'Cohort'],
             ['Dx', 'Dx3'], 
             ['Dx', 'Subtype'],  
             ['Dx', 'Dx3', 'Subtype']]


Ks = gmetrics.Density.unique()
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>Kmin]])

for Kmin, Kmax in K_ranges:
    AUC = gmetrics[(gmetrics.Density >= Kmin) & (gmetrics.Density <= Kmax)].groupby('SubjID').sum()
    AUC = AUC.merge(demographics, left_index=True, right_index=True)
    
    group_stats = [AUC.groupby(groups).agg(stat_list).stack().reset_index().rename(columns={f'level_{len(groups)}':'Stat'})
                   for i, groups in enumerate(contrasts)]
    
    for i, data in enumerate(group_stats):
        data.sort_values(contrasts[i] + ['Stat'])
        data['Contrast'] = i

    aggregation_table = pd.concat(group_stats).set_index(['Contrast', 'Dx', 'Dx3', 'Subtype', 'Cohort']).reset_index()
    aggregation_table['Kmin'] = Kmin
    aggregation_table['Kmax'] = Kmax
    
    aggregation_table.to_csv(f"{output_dir}/Group_stats_K{Kmin}-{Kmax}.csv")
    
    

logging.debug(f"\n\nLocal variables:{list(locals())}")

logging.debug(f"\n\nOUTPUT TABLES GENERATED")

    
