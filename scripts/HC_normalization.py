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


# Normalize on healthy subjects
thickness_volume = pd.read_csv(f"{data_dir}/CT_Volume.csv", index_col="SubjID")
demographics = pd.read_csv(f"{data_dir}/Demographics.csv", index_col="SubjID")
residuals = pd.read_csv(f"{data_dir}/Data_residuals.csv", index_col="SubjID")
residuals = thickness_volume.merge(demographics['Dx'], left_index=True, right_index=True)

mu = residuals.groupby('Dx').mean()
sd = residuals.groupby('Dx').std()

zscores = ((residuals - mu.loc[0]) / sd.loc[0]).drop('Dx', axis=1)


zscores.to_csv(f"{data_dir}/Data_zscores.csv")

logging.debug(f"DATA ZSCORED\n{log_func(locals())}\n\n")
