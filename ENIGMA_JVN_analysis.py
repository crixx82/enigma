# Input data paths

data_dir = "./data"
covariates = f"{data_dir}/Covariates_simulation.csv"
thickness = f"{data_dir}/CorticalMeasuresENIGMA_ThickAvg.csv"
volume = f"{data_dir}/SubcorticalMeasuresENIGMA_VolAvg.csv"
output_dir = f"./output"

nj = -1

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
import argparse

logging.basicConfig(filename=f"{output_dir}/log.txt",
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S %Z')
existing = []
def log_func(existing, var_dict):
	new = {key:type(value) for key, value in var_dict.items() if key not in existing}
	existing.extend([key for key in var_dict if key not in existing])
	return new, existing
#-----------------------------------------------------------------------------------------------------------------


# Assemble dataframe for analyses and screen subjects

demographics = pd.read_csv(covariates, index_col="SubjID")
dashless_columns = pd.Series(demographics.columns.str.split('_')).apply(lambda x: ''.join(str.capitalize(s) for s in x))
demographics.rename(columns=dict(zip(demographics.columns, dashless_columns)), inplace=True)

demographics.loc[demographics.Subtype.isna(), 'Subtype'] = 0
demographics['Minor'] = np.int32(demographics.Age<18.)
demographics['Durill3'] = np.int32(demographics.Durill<3.)
demographics['Age2'] = demographics['Age']**2

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

demographics.to_csv(f"{data_dir}/Demographics.csv")
thickness_volume.to_csv(f"{data_dir}/CT_Volume.csv")

logging.debug(f"DATAFRAME CREATED\n{log_func(existing, locals())[0]}\n\n")
#-----------------------------------------------------------------------------------------------------------------


# Correct for covariates

def covar_correct(X, Y, data):
    
    def get_resid(X, y, data):
        r = data[y] - LinearRegression(n_jobs=1).fit(data[X], data[y]).predict(data[X])
        return r
    
    R = Parallel(n_jobs=nj)(delayed(get_resid)(X, y, data) for y in Y)
    return np.asanyarray(R).T

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
thickness_volume = pd.read_csv(f"{data_dir}/CT_Volume.csv", index_col="SubjID")
brain_regions = thickness_volume.columns                               
                               
data = demographics[['Age', 'Hand']].merge(thickness_volume, left_index=True, right_index=True)
                               
X = ['Age', 'Hand']
Y = brain_regions

residuals = thickness_volume.copy()
residuals[brain_regions] = covar_correct(X, Y, data)


residuals.to_csv(f"{data_dir}/Data_residuals.csv")

logging.debug(f"DATA CORRECTED FOR CONFOUNDS\n{log_func(existing, locals())[0]}\n\n")

#-----------------------------------------------------------------------------------------------------------------


# Normalize on healthy subjects

thickness_volume = pd.read_csv(f"{data_dir}/CT_Volume.csv", index_col="SubjID")
demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
residuals = pd.read_csv(f"{data_dir}/Data_residuals.csv", index_col="SubjID")
residuals = thickness_volume.merge(demographics['Dx'], left_index=True, right_index=True)

mu = residuals.groupby('Dx').mean()
sd = residuals.groupby('Dx').std()

zscores = ((residuals - mu.loc[0]) / sd.loc[0]).drop('Dx', axis=1)


zscores.to_csv(f"{data_dir}/Data_zscores.csv")

logging.debug(f"DATA ZSCORED\n{log_func(existing, locals())[0]}\n\n")
#-----------------------------------------------------------------------------------------------------------------


# Generate graphs and extract metrics

thresholds = np.arange(95,101, 1)

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

logging.debug(f"GRAPH METRICS COMPUTED\n{log_func(existing, locals())[0]}\n\n")
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



Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 1)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+3]])

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
        
        
logging.debug(f"GROUP STATS COMPUTED\n{log_func(existing, locals())[0]}\n\n")

#-----------------------------------------------------------------------------------------------------------------


# Correlation coefficients

demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
gmetrics = pd.read_csv(f"{output_dir}/Graph_metrics.csv", index_col="SubjID")
metrics = gmetrics.columns.to_list()


Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 1)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+3]])

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
        
        
logging.debug(f"CORRELATIONS COMPUTED\n{log_func(existing, locals())[0]}\n\n")
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

Ks = np.arange(gmetrics.Density.min(), gmetrics.Density.max()+1, 1)
K_ranges = np.array([(Kmin, Kmax) for Kmin in Ks for Kmax in Ks[Ks>=Kmin+3]])

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
        
        
logging.debug(f"REGRESSION MODELS COMPUTED\n{log_func(existing, locals())[0]}\n\n\n\n")
#-----------------------------------------------------------------------------------------------------------------
