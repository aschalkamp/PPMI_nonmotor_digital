import pandas as pd
import dask.dataframe as dd
import numpy as np
from scipy import stats

from importlib import reload
from functools import reduce
import datetime

import sys
sys.path.insert(1,'../scripts')
import utils

from tsfresh import extract_features, extract_relevant_features, select_features, feature_selection
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_extraction
from tsfresh.transformers import RelevantFeatureAugmenter

import pickle

data_path = '/scratch/c.c21013066/data/ppmi'
path = '/scratch/c.c21013066/data/ppmi/accelerometer'

demo = pd.read_csv(f'{data_path}/phenotypes2021/demographics_clean.csv',parse_dates=['date_birth'])

# load smartwatch data
reload(utils)
merged, df = utils.load_specific_timeseries(demo,path,names=[sys.argv[1]])

df_clean = utils.clean_timeseries(df)
predictor = df_clean.filter(regex='(walking|step|efficiency|total_sleep|pulse|deep|light|rem|nrem|rmssd|wake)').columns
predictor_nan = df_clean[predictor].isna().sum() > 0
#ddf = dd.from_pandas(step_clean, npartitions=10)
print(predictor)
extraction_settings = feature_extraction.ComprehensiveFCParameters()

X = extract_features(df_clean[np.hstack(['participant','date_local_adj',predictor[~predictor_nan]])], column_id='participant', column_sort='date_local_adj',
                     default_fc_parameters=extraction_settings,
                     # we impute = remove all NaN features automatically
                     impute_function=impute,n_jobs=4)
for p in predictor[predictor_nan]:
    print(p)
    df_nona = df_clean.dropna(subset=[p])

    X_ = extract_features(df_nona[np.hstack(['participant','date_local_adj',p])], column_id='participant', column_sort='date_local_adj',
                     default_fc_parameters=extraction_settings,
                     # we impute = remove all NaN features automatically
                     impute_function=impute,n_jobs=4)
    X = pd.merge(X,X_,right_index=True,left_index=True,how='outer')

X.to_csv(f'{path}/extracted_features/{sys.argv[1]}.csv')