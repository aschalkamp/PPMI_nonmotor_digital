# -*- coding: utf-8 -*-
"""
Small helper functions for preprocessing messy data
"""

import pandas as pd
import datetime
import numpy as np
from functools import reduce

from sklearn import preprocessing, linear_model


def intersect_subjects(*inputs):
    """
    Finds intersection of participants between `inputs`
    Parameters
    ----------
    inputs : pandas.DataFrame
        Index of dataframes will be used as identifier
    Returns
    -------
    outputs : (N,) list of pandas.DataFrame
        Input dataframes with only shared participants retained
    """

    # get intersection of subjects (row indices of input dataframes)
    subjects = list(set.intersection(*[set(f.index) for f in inputs]))

    # subset dataframes with overlapping subjects
    outputs = [f[f.index.isin(subjects)].copy().sort_index() for f in inputs]

    return outputs


def clean_data(*inputs, cutoff=0.8,axis="both"):
    """
    Parameters
    ----------
    inputs : pandas.DataFrame
        Input dataframes (all same length!)
    cutoff : (0, 1) float, optional
        Percent of subjects/variables that must have non-NA values to be
        retained. Default: 0.8
    axis : rows, columns, both
    Returns
    -------
    inputs : list of pd.core.frame.DataFrame
        Cleaned inputs
    """

    if cutoff < 0 or cutoff > 1:
        raise ValueError(f'Supplied cutoff {cutoff} not between [0, 1].')
    if len(inputs) > 1:
        inputs = intersect_subjects(*inputs)
    else:
        inputs = inputs[0]
    if axis in ['both','columns']:
        inputs = [f.dropna(thresh=cutoff * f.shape[0], axis=1) for f in inputs]
    if axis in ['both','rows']:
        inputs = [f.dropna(thresh=cutoff * f.shape[1], axis=0) for f in inputs]

    inputs = intersect_subjects(*inputs)

    return inputs


def impute_data(inputs, strategy='median'):
    """
    Imputes missing data in ``inputs``
    Parameters
    ----------
    inputs : list of pandas.DataFrame
        List of input data
    strategy : {'mean', 'median'}, optional
        Imputation strategy. Default: 'median'
    Returns
    -------
    ouputs : list of pandas.DataFrame
        List of imputed data
    """

    from sklearn.impute import SimpleImputer

    # https://github.com/scikit-learn/scikit-learn/pull/9212
    impute = SimpleImputer(strategy=strategy)
    outputs = [pd.DataFrame(impute.fit_transform(f[f.columns]), columns=f.columns,
                            index=f.index) for f in inputs]

    return outputs

def get_visit(df, participants, visit='SC', cutoff=0.8):
    """
    Extracts specified `visit` and `participants` from `df`
    Parameters
    ----------
    df : pd.DataFrame
        "Raw" dataframe that needs a bit of love and attention
    participants : list-of-int
        List of participant IDs to retain in `df`
    visit : str, optional
        Visit to retain in `df`. Default: 'SC'
    cutoff : (0, 1) float, optional
        Percent of subjects/variables that must have non-NA values to be
        retained. Default: 0.8
    Returns
    -------
    df : pd.DataFrame
        Provided data frame after receiving some love and attention
    date : pd.DataFrame
        Visit date extracted from `df`, with same index as `df`
    """

    df = df.dropna(subset=['visit', 'date']) \
           .query(f'participant in {participants} & visit == "{visit}"') \
           .drop('visit', axis=1) \
           .set_index('participant')
    df = df.dropna(axis=1, thresh=(cutoff * len(df)))
    date = pd.DataFrame(df.pop('date'))

    return df, date

def get_visit_raw(df, participants, visit='SC'):
    """
    Extracts specified `visit` and `participants` from `df`
    Parameters
    ----------
    df : pd.DataFrame
        "Raw" dataframe that needs a bit of love and attention
    participants : list-of-int
        List of participant IDs to retain in `df`
    visit : str, optional
        Visit to retain in `df`. Default: 'SC'
    Returns
    -------
    df : pd.DataFrame
        Provided data frame after receiving some love and attention

    """

    df = df.dropna(subset=['visit', 'date']) \
           .query(f'participant in {participants} & visit == "{visit}"') \
           .drop('visit', axis=1) \
           .set_index('participant')
    #date = pd.DataFrame(df.pop('date'))

    return df

def get_visit_age(df):
    """
    From birthdate and visit date information calculate age of subject at visit
    ---------
    df : pd.DataFrame
    Returns
    -------
    df : pd.DataFrame
        with added column for session age
    """
    # session age is session date - birthdate
    df['visit_age'] = (df['date'] - df['date_birth']) / np.timedelta64(1,'Y')
    return df

def get_visit_age_adni(df):
    """
    From age at first visit and visit date information calculate age of subject at visit
    ---------
    df : pd.DataFrame
    Returns
    -------
    df : pd.DataFrame
        with added column for session age
    """
    # infer birthdate
    df['visit_age'] = df['bl_age'] + df['time']
    return df

def load_longitudinal_data(participants,behavior):
    """
    Loads longitudinal behavioral data
    Parameters
    ----------
    participants : list
        List of participant IDs for whom behavioral data should be loaded
    behavior: pd.DataFrame
    Returns
    -------
    behavior : pandas.DataFrame
        Longitudinal behavioral data for `participants`
    """

    # ensure this is a list (DataFrame.query doesn't work with arrays)
    participants = list(participants)

    # load ALL behavioral data (not just first visit!) and drop duplicates
    #first = (behavior.drop_duplicates(['participant', 'visit'], 'first')
    #                 .reset_index(drop=True))
    #last = (behavior.drop_duplicates(['participant', 'visit'], 'last')
    #                .reset_index(drop=True))
    #behavior = first.combine_first(last)
    behavior = behavior.query(f'participant in {participants}')

    # generate a continuous "time" variable (visit_date - base_visit_date)
    base = behavior.groupby('participant')['date'].min().rename('base')
    behavior = pd.merge(behavior, base, on='participant')
    time = (behavior['date'] - behavior['base']) / np.timedelta64(1, 'Y')
    behavior = behavior.assign(time=time).set_index('participant')
    return behavior

def recode(behavior):
    """
    Transform clinical tests such that higher values indicate worse performance
    """
    import sys
    sys.path.insert(1, '/scratch/c.c21013066/PPMI_DataPreparation/phenotype')
    from _thresholds import BEHAVIORAL_INFO as RECODE
    
    # find intersection of columns (so data we have and were also metadata available)
    overlap = np.intersect1d(behavior.columns,list(RECODE.keys()))
    print(overlap)
    recoded = behavior.copy(deep=True)
    for key in overlap:
        item = RECODE[key]
        if item['recode']:
            old = recoded[key]
            recoded[key] = item['max'] - recoded[key]
            recoded.loc[old.isna(),key] = np.nan
    return recoded

def get_mean_decline(df,score):
    first = df.groupby('participant').first()
    last = df.groupby('participant').last()
    decline = (last[score]-first[score]) / ((last['date'] - first['date']) / np.timedelta64(1,"Y"))
    return decline

def date_to_datetime(df):
    dates = df.filter(regex='^date').columns
    df[dates] = df[dates].apply(pd.to_datetime)
    return df

def prepare_demographics(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age']):
    
    demographics = pd.read_csv(f'{path}/demographics_clean.csv',index_col=0).set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    # only keep PD, HC
    demographics = demographics[np.logical_or(demographics['diagnosis']=='pd',demographics['diagnosis']=='hc')]
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {demographics.shape}')
    demographics = clean_data([demographics],axis='rows',cutoff=0.5)[0]
    demographics['diagnosis_bl_age'] = demographics['age'] - demographics['diagnosis_age']
    return demographics

def merge_SC_BL(demographics,behavior):
    # merge BL and SC
    # behavioral data acquisition was split across screening + baseline visits
    # so we need to take the earliest visit for each measure
    # that is, not all measures were collected at screening so we need to use
    # the baseline visit scores for those measures
    # unfortunately which visit various measures were initially collected at
    # DIFFERED for PD and HC individuals, so we need to do this separately for
    # the two groups and then merge them back together
    #beh, beh_dates = [], []
    #for diagnosis in np.unique(demographics['diagnosis']):
    #    participants = demographics.query(f'diagnosis == "{diagnosis}"').index
    #    beh_sc, beh_date = get_visit(behavior, list(participants), visit='SC')
    #    beh_bl, _ = get_visit(behavior, list(participants), visit='BL')
    #    drop = np.intersect1d(beh_sc.columns, beh_bl.columns)
    #    beh += [pd.merge(beh_sc, beh_bl.drop(drop, axis=1), on='participant')]
    #    beh_dates += [beh_date]
    # OR: JUST FILL IN ALL MISSING VALUES FROM SCREENING WITH VALUES FROM BASELINE
    participants = demographics.index
    beh_sc = get_visit_raw(behavior, list(participants), visit='SC').reset_index()
    beh_bl = get_visit_raw(behavior, list(participants), visit='BL').reset_index()
    beh = beh_sc.append(beh_bl)
    first = (beh.drop_duplicates(['participant'], 'first')
                     .reset_index(drop=True))
    last = (beh.drop_duplicates(['participant'], 'last')
                    .reset_index(drop=True))
    behavior_df_bl = first.combine_first(last).set_index('participant')
    #beh_sc = beh_sc.fillna(beh_bl)
    #beh_date = pd.concat([sc_date,beh_date], join='inner')
    #behavior_df_bl = pd.concat(beh, join='inner')
    #beh_date = pd.concat(beh_dates, join='inner')
    #behavior_df_bl = pd.merge(behavior_df_bl,beh_date,on='participant')
    #behavior_df_bl = pd.merge(behavio,beh_date,on='participant')
    behavior_df_bl['visit'] = 'BL'
    behavior_df = behavior[~np.logical_or(behavior['visit']=='SC',behavior['visit']=='BL')]
    behavior_df = behavior_df.append(behavior_df_bl.reset_index())
    behavior_df = behavior_df.set_index(['participant','date']).sort_index()
    return behavior_df.reset_index()

def merge_visitID_otherdates(behavior,key='visit'):
    # merge visits with other date but same ID
    # OR: FILL IN ALL MISSING VALUES FROM FIRST visit with SECOND visit
    behavior = behavior.set_index(['participant',key])
    duplicated = behavior[behavior.index.duplicated(keep=False)]
    behavior_df = behavior.drop(duplicated.index,axis=0)
    behavior_df = behavior_df.reset_index()
    duplicated = duplicated.reset_index()
    print(f'this many duplicated entries: {duplicated.shape}')
    first = (duplicated.drop_duplicates(['participant',key], 'first')
                     .reset_index(drop=True))
    print(first.shape)
    last = (duplicated.drop_duplicates(['participant',key], 'last')
                    .reset_index(drop=True))
    print(last.shape)
    behavior_df_bl = last.combine_first(first).set_index('participant')
    print(behavior_df_bl.shape)
    #beh_sc = beh_sc.fillna(beh_bl)
    #beh_date = pd.concat([sc_date,beh_date], join='inner')
    #behavior_df_bl = pd.concat(beh, join='inner')
    #beh_date = pd.concat(beh_dates, join='inner')
    #behavior_df_bl = pd.merge(behavior_df_bl,beh_date,on='participant')
    #behavior_df_bl = pd.merge(behavio,beh_date,on='participant')
    #behavior_df_bl['visit'] = first['visit']
    behavior_df = behavior_df.append(behavior_df_bl.reset_index())
    behavior_df = behavior_df.set_index(['participant','date']).sort_index()
    return behavior_df.reset_index()

def match_nearest(df,col1,col2,merge='date'):
    df1 = df[np.hstack([col1,[merge]])].set_index(merge).sort_index().dropna(axis='rows',how='all')
    df2 = df[np.hstack([col2,[merge]])].set_index(merge).sort_index().dropna(axis='rows',how='all')
    data = df.drop(columns=col2).set_index(merge).dropna(axis='rows',how='all')
    df2 = df2.reindex(df1.index,method='nearest',tolerance=pd.Timedelta(365,'days'))
    return pd.merge(df2,data,right_index=True,left_index=True).reset_index()

def match_only_nearest(df,col1,col2,merge='date',tolerance=pd.Timedelta(80,'days')):
    df1 = df[np.hstack([col1,[merge]])].set_index(merge).sort_index().dropna(axis='rows',how='all')
    df2 = df[np.hstack([col2,[merge]])].set_index(merge).sort_index().dropna(axis='rows',how='all')
    data = df.drop(columns=col2).set_index(merge).sort_index().dropna(axis='rows',how='all',subset=col1)
    return pd.merge_asof(df2, data, right_index=True,left_index=True, direction="nearest",tolerance=tolerance).reset_index()

def match_only_nearest_df(df1,df2,merge='date',tolerance=pd.Timedelta(80,'days'),suffixes=['_x','_y']):
    df1 = df1.set_index(merge).sort_index().dropna(axis='rows',how='all')
    df2 = df2.set_index(merge).sort_index().dropna(axis='rows',how='all')
    return pd.merge_asof(df1, df2, by='participant',right_index=True,left_index=True, direction="nearest",tolerance=tolerance,suffixes=suffixes).reset_index()

def combine_phenotypes(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['putamen_l','putamen_r','caudate_l','caudate_r'],tolerance=pd.Timedelta(365,'days')):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes/{file}.csv")[np.hstack([clinical,['date','participant','visit']])]
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes/biospecimen_clean.csv")[np.hstack([bio,['date','participant','visit']])]
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes/datscan_clean.csv")[np.hstack([dat,['date','participant','visit']])]
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes/demographics_clean.csv',index_col=0).set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    # only keep PD, HC
    demographics = demographics[np.logical_or(demographics['diagnosis']=='pd',demographics['diagnosis']=='hc')]
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    # grab longitudinal data for behavior for all participants where we know demographics and merge on date
    longitudinal = [behavior.drop(columns=['visit']),datscan.drop(columns=['visit']),biospecimen.drop(columns=['visit'])]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['participant','date'],
                                            how='outer'), longitudinal)
    print(df_merged.shape)
    df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows').set_index('participant')
    print(f'drop visits with neither clinical, biospecimen nor datscan data {df_merged.shape}')
    df_merged = df_merged.groupby('participant').apply(match_nearest,col1=clinical,col2=dat,merge='date',tolerance=tolerance)
    df_merged = df_merged.groupby('participant').apply(match_nearest,col1=clinical,col2=bio,merge='date',tolerance=tolerance)
    df_merged = load_longitudinal_data(demographics.index,df_merged.reset_index())
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age(df_merged)
    df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    bl_age = df_merged.groupby('participant').first()['visit_age'].rename('bl_age')
    df_merged = pd.merge(df_merged,bl_age,on='participant')
    df_merged['diagnosis_bl_age'] = df_merged['bl_age'] - df_merged['diagnosis_age']
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def combine_phenotypes_adni(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['ventricles','hippocampus','wholebrain','entorhinal','fusiform','midtemporal',
                                                                         'intracerebroventricular']):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes/{file}.csv")[np.hstack([clinical,['date','participant','visit']])]
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes/biospecimen_clean.csv")[np.hstack([bio,['date','participant','visit']])]
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes/imaging_clean.csv")[np.hstack([dat,['date','participant','visit']])]
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes/demographics_clean.csv',index_col=0).set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['Male','Female'],[1,0])
    # grab longitudinal data for behavior for all participants where we know demographics and merge on date
    longitudinal = [behavior.drop(columns=['visit']),datscan.drop(columns=['visit']),biospecimen.drop(columns=['visit'])]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['participant','date'],
                                            how='outer'), longitudinal)
    print(df_merged.shape)
    df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows').set_index('participant')
    print(f'drop visits with neither clinical, biospecimen nor imaging data {df_merged.shape}')
    df_merged = df_merged.groupby('participant').apply(match_nearest,col1=clinical,col2=dat,merge='date')
    df_merged = df_merged.groupby('participant').apply(match_nearest,col1=clinical,col2=bio,merge='date')
    df_merged = load_longitudinal_data(demographics.index,df_merged.reset_index())
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age_adni(df_merged)
    df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def combine_phenotypes_only_closest(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['putamen_l','putamen_r','caudate_l','caudate_r']):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes/{file}.csv")[np.hstack([clinical,['date','participant','visit']])]
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes/biospecimen_clean.csv")[np.hstack([bio,['date','participant','visit']])]
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes/datscan_clean.csv")[np.hstack([dat,['date','participant','visit']])]
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes/demographics_clean.csv',index_col=0).set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    # merge BL and SC visits
    behavior = merge_SC_BL(demographics,behavior)
    # only keep PD, HC
    demographics = demographics[np.logical_or(demographics['diagnosis']=='pd',demographics['diagnosis']=='hc')]
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    # grab longitudinal data for behavior for all participants where we know demographics and merge on date
    longitudinal = [behavior.drop(columns=['visit']),datscan.drop(columns=['visit']),biospecimen.drop(columns=['visit'])]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['participant','date'],
                                            how='outer'), longitudinal)
    print(df_merged.shape)
    df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows').set_index('participant')
    print(f'drop visits with neither clinical, biospecimen nor datscan data {df_merged.shape}')
#     df_merged_keep = df_merged.groupby('participant').apply(match_only_nearest,col1=dat,col2=clinical,merge='date')
#     df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col1=clinical,col2=dat,merge='date')
#     df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[dat].isna().any(axis=1)]).sort_index()
#     df_merged_keep = df_merged.groupby('participant').apply(match_only_nearest,col1=bio,col2=np.hstack([clinical,dat]),merge='date')
#     df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col1=np.hstack([clinical,dat]),col2=bio,merge='date')
#     df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[bio].isna().any(axis=1)]).sort_index()
    df_merged = df_merged.groupby('participant').apply(match_only_nearest,col1=dat,col2=np.hstack([clinical,bio]),merge='date',
                                                       tolerance=pd.Timedelta(60,'days'))
    #df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col2=clinical,col1=dat,merge='date')
    #df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[clinical].isna().any(axis=1)]).sort_index()
    df_merged = df_merged.groupby('participant').apply(match_only_nearest,col1=bio,col2=np.hstack([clinical,dat]),merge='date',
                                                      tolerance=pd.Timedelta(60,'days'))
    #df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col2=np.hstack([clinical,dat]),col1=bio,merge='date')
    #df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[np.hstack([clinical,dat])].isna().any(axis=1)]).sort_index()
    df_merged = load_longitudinal_data(demographics.index,df_merged.reset_index())
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age(df_merged)
    df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    bl_age = df_merged.groupby('participant').first()['visit_age'].rename('bl_age')
    df_merged = pd.merge(df_merged,bl_age,on='participant')
    df_merged['diagnosis_bl_age'] = df_merged['bl_age'] - df_merged['diagnosis_age']
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def combine_phenotypes_to_datscan(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['putamen_l','putamen_r','caudate_l','caudate_r']):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes/{file}.csv")[np.hstack([clinical,['date','participant','visit']])]
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes/biospecimen_clean.csv")[np.hstack([bio,['date','participant','visit']])]
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes/datscan_clean.csv")[np.hstack([dat,['date','participant','visit']])]
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes/demographics_clean.csv',index_col=0).set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    # merge BL and SC visits
    behavior = merge_SC_BL(demographics,behavior)
    # only keep PD, HC
    demographics = demographics[np.logical_or(np.logical_or(demographics['diagnosis']=='pd',demographics['diagnosis']=='hc'),
                                              demographics['diagnosis']=='swedd')]
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    # grab longitudinal data for behavior for all participants where we know demographics and merge on date
    longitudinal = [behavior.drop(columns=['visit']),datscan.drop(columns=['visit']),biospecimen.drop(columns=['visit'])]
    #df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['participant','date'],
    #                                        how='outer'), longitudinal)
    df_merged = match_only_nearest_df(longitudinal[1],longitudinal[0],merge='date',tolerance=pd.Timedelta(182,'days'))
    df_merged = match_only_nearest_df(df_merged,longitudinal[2],merge='date',tolerance=pd.Timedelta(182,'days'))
    #print(df_merged.shape)
    #df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows').set_index('participant')
    print(f'drop visits with neither clinical, biospecimen nor datscan data {df_merged.shape}')
#     df_merged_keep = df_merged.groupby('participant').apply(match_only_nearest,col1=dat,col2=clinical,merge='date')
#     df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col1=clinical,col2=dat,merge='date')
#     df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[dat].isna().any(axis=1)]).sort_index()
#     df_merged_keep = df_merged.groupby('participant').apply(match_only_nearest,col1=bio,col2=np.hstack([clinical,dat]),merge='date')
#     df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col1=np.hstack([clinical,dat]),col2=bio,merge='date')
#     df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[bio].isna().any(axis=1)]).sort_index()
    #df_merged = df_merged.groupby('participant').apply(match_only_nearest,col2=dat,col1=clinical,merge='date',
    #                                                   tolerance=pd.Timedelta(80,'days'))
    #df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col2=clinical,col1=dat,merge='date')
    #df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[clinical].isna().any(axis=1)]).sort_index()
    #df_merged = df_merged.groupby('participant').apply(match_only_nearest,col1=bio,col2=np.hstack([clinical,dat]),merge='date',
    #                                                  tolerance=pd.Timedelta(80,'days'))
    #df_merged_drop = df_merged.groupby('participant').apply(match_only_nearest,col2=np.hstack([clinical,dat]),col1=bio,merge='date')
    #df_merged = df_merged_keep.append(df_merged_drop[df_merged_drop[np.hstack([clinical,dat])].isna().any(axis=1)]).sort_index()
    df_merged = load_longitudinal_data(demographics.index,df_merged.reset_index())
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age(df_merged)
    df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    bl_age = df_merged.groupby('participant').first()['visit_age'].rename('bl_age')
    df_merged = pd.merge(df_merged,bl_age,on='participant')
    df_merged['diagnosis_bl_age'] = df_merged['bl_age'] - df_merged['diagnosis_age']
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def combine_phenotypes_only_closest_final(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['putamen_l','putamen_r','caudate_l','caudate_r']):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes2023/{file}.csv")[np.hstack([clinical,['date','participant','visit']])].set_index('participant')
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes2023/biospecimen_clean.csv")[np.hstack([bio,['date','participant']])].set_index('participant')
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes2023/datscan_clean.csv")[np.hstack([dat,['date','participant']])].set_index('participant')
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes2023/demographics_clean.csv').set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    print('demographics',demographics.shape)
    # merge BL and SC visits
    behavior = merge_SC_BL(demographics,behavior.reset_index()).set_index('participant')
    print('behavior',behavior.shape)
    print('behavior clean',behavior[clinical].dropna(subset=clinical,axis='rows',how='all').shape)
    # combine data
    datscan = datscan.reset_index().set_index(['date']).sort_index().reset_index()
    biospecimen = biospecimen.reset_index().set_index(['date']).sort_index().reset_index()
    behavior = behavior.reset_index().set_index(['date']).sort_index().reset_index()
    merged = pd.merge_asof(behavior,datscan,by='participant',on='date',direction='nearest',tolerance=pd.Timedelta(80,'days'))
    print('behdat',merged.shape)
    print('beh in dat',merged[clinical].dropna(subset=clinical,how='all',axis='rows').shape)
    df_merged = pd.merge_asof(merged,biospecimen,by='participant',on='date',direction='nearest',tolerance=pd.Timedelta(80,'days'))
    print(df_merged.shape)
    print('beh in dat',df_merged[clinical].dropna(subset=clinical,how='all',axis='rows').shape)
    df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows')
    # add derived covariates
    df_merged = load_longitudinal_data(demographics.index,df_merged)
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age(df_merged)
    #df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    #print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    bl_age = df_merged.groupby('participant').first()['visit_age'].rename('bl_age')
    df_merged = pd.merge(df_merged,bl_age,on='participant')
    df_merged['diagnosis_bl_age'] = df_merged['bl_age'] - df_merged['diagnosis_age']
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date','date_diagnosis']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def combine_phenotypes_only_closest_final_old(file,path,covs=['gender','education','diagnosis_bl_age','time','bl_age'],
               clinical=['updrs_i','moca','epworth','semantic_fluency','symbol_digit','lns'],
              bio=['ptau','ttau','abeta_1-42','csf_alpha-synuclein'],dat=['putamen_l','putamen_r','caudate_l','caudate_r']):
    
    # clinical and demographics
    behavior = pd.read_csv(f"{path}/phenotypes/{file}.csv")[np.hstack([clinical,['date','participant','visit']])].set_index('participant')
    behavior = date_to_datetime(behavior)
    biospecimen = pd.read_csv(f"{path}/phenotypes/biospecimen_clean.csv")[np.hstack([bio,['date','participant']])].set_index('participant')
    biospecimen = date_to_datetime(biospecimen)
    datscan = pd.read_csv(f"{path}/phenotypes/datscan_clean.csv")[np.hstack([dat,['date','participant']])].set_index('participant')
    datscan = date_to_datetime(datscan)
    demographics = pd.read_csv(f'{path}/phenotypes/demographics_clean.csv').set_index('participant')
    demographics = date_to_datetime(demographics)
    demographics['gender'] = demographics['gender'].replace(['m','f'],[1,0])
    demographics['diagnosis_age'] = (demographics['date_diagnosis'] - demographics['date_birth']) / np.timedelta64(1,'Y')
    print('demographics',demographics.shape)
    # merge BL and SC visits
    behavior = merge_SC_BL(demographics,behavior.reset_index()).set_index('participant')
    print('behavior',behavior.shape)
    print('behavior clean',behavior[clinical].dropna(subset=clinical,axis='rows',how='all').shape)
    # combine data
    datscan = datscan.reset_index().set_index(['date']).sort_index().reset_index()
    biospecimen = biospecimen.reset_index().set_index(['date']).sort_index().reset_index()
    behavior = behavior.reset_index().set_index(['date']).sort_index().reset_index()
    merged = pd.merge_asof(behavior,datscan,by='participant',on='date',direction='nearest',tolerance=pd.Timedelta(80,'days'))
    print('behdat',merged.shape)
    print('beh in dat',merged[clinical].dropna(subset=clinical,how='all',axis='rows').shape)
    df_merged = pd.merge_asof(merged,biospecimen,by='participant',on='date',direction='nearest',tolerance=pd.Timedelta(80,'days'))
    print(df_merged.shape)
    print('beh in dat',df_merged[clinical].dropna(subset=clinical,how='all',axis='rows').shape)
    df_merged = df_merged.dropna(subset=np.hstack([clinical,bio,dat]),how='all',axis='rows')
    # add derived covariates
    df_merged = load_longitudinal_data(demographics.index,df_merged)
    print(f'drop participants where no demographic info available and combine nearby dates to clinical visit dates {df_merged.shape}')
    # concat with demographics which serve as covariates (age, education, sex,date_diagnosis)
    df_merged = pd.merge(df_merged,demographics,right_index=True,left_index=True)
    print(df_merged.shape)
    df_merged = get_visit_age(df_merged)
    #df_merged = clean_data([df_merged],axis='rows',cutoff=0.5)[0]
    #print(df_merged.shape)
    #behavior_df = clean_data([behavior_df],axis='columns',cutoff=0.8)[0]
    #df_merged[np.hstack([clinical,bio,dat])] = .impute_data([df_merged[np.hstack([clinical,bio,dat])]])[0]
    bl_age = df_merged.groupby('participant').first()['visit_age'].rename('bl_age')
    df_merged = pd.merge(df_merged,bl_age,on='participant')
    df_merged['diagnosis_bl_age'] = df_merged['bl_age'] - df_merged['diagnosis_age']
    n_visits = df_merged.groupby('participant').size().rename('n_visits')
    df_merged = pd.merge(df_merged,n_visits,on='participant')[np.hstack([covs,clinical,bio,dat,['n_visits','date','date_diagnosis']])]
    # only keep with at least 2 visits
    #df_merged = df_merged[behavior_df['n_visits']>1]
    # recode such that higher values mean worse performance
    df_merged[clinical] = recode(df_merged[clinical])
    return df_merged

def get_DatScan_IDPs(df):
    '''Derive from SBR more measures of interest'''
    df["datscan_caudate_mean"] = df[['datscan_caudate_l','datscan_caudate_r']].mean(axis=1)
    df["datscan_putamen_mean"] = df[['datscan_putamen_l','datscan_putamen_r']].mean(axis=1)
    df["datscan_mean"] = df[['datscan_putamen_l','datscan_putamen_r','datscan_caudate_l','datscan_caudate_r']].mean(axis=1)
    df["datscan_left_mean"] = df[['datscan_putamen_l','datscan_caudate_l']].mean(axis=1)
    df["datscan_right_mean"] = df[['datscan_putamen_r','datscan_caudate_r']].mean(axis=1)
    df['datscan_count_density'] = df['datscan_caudate_mean']/df['datscan_putamen_mean']
    #100 * [(left - right) / mean(left + right)]
    df['datscan_asymmetry'] = np.abs(100 * ((df['datscan_left_mean'] - df['datscan_right_mean']) / df[['datscan_left_mean','datscan_right_mean']].mean(axis=1)))
    df['datscan_sai'] = 100 * 2 * (np.abs(df['datscan_left_mean'] - df['datscan_right_mean']) / (df['datscan_left_mean'] + df['datscan_right_mean']))
    df['datscan_caudate_asymmetry'] = 2* np.abs(100*((df['datscan_caudate_l'] - df['datscan_caudate_r']) / (df['datscan_caudate_l']+df['datscan_caudate_r'])))
    df['datscan_putamen_asymmetry'] = 2* np.abs(100*((df['datscan_putamen_l'] - df['datscan_putamen_r']) / (df['datscan_putamen_l']+df['datscan_putamen_r'])))
    return df

def get_DAT_deficit(datscan,age_sex_correct='group'):
    """Derive from quantitative datscan whether deficit or not
    dat deficit: minimum putamen specific binding ratio (SBR) and <65% age/sex-expected lowest putamen SBR was used as a cut-off for DAT deficit"""
    datscan['putamen_min'] = datscan[['datscan_putamen_l','datscan_putamen_r']].min(axis=1)
    if age_sex_correct=='group':
        datscan['age_group'] = pd.cut(datscan['visit_age'], bins=np.arange((datscan['visit_age'].min()//10)*10, (datscan['visit_age'].max()//10)*10+20, 10))
        hc = datscan[datscan['diagnosis']=='hc']
        # get group and age average
        #thresh = hc.groupby(['age_group','gender'])['putamen_min'].quantile(.35)
        thresh = hc.groupby(['age_group','gender'])['putamen_min'].mean() - 2*hc.groupby(['age_group','gender'])['putamen_min'].std()
        print(hc.groupby(['age_group','gender']).size())
        print(thresh)
        datscan['dat_deficit'] = datscan.apply(lambda row: row['putamen_min'] < thresh.loc[(row['age_group'], row['gender'])], axis=1)
        return datscan
    elif age_sex_correct=='regression':
        hc = datscan[datscan['diagnosis']=='hc']
        model = linear_model.LinearRegression()
        model.fit(hc[['visit_age']],hc['putamen_min'])
        print(model.coef_)
        age_coeff = model.coef_[0]
        datscan['putamen_min_age_corr'] = datscan['putamen_min'] - age_coeff*datscan['visit_age']
        hc = datscan[datscan['diagnosis']=='hc']
        thresh = hc['putamen_min_age_corr'].quantile(.35)
        thresh = hc['putamen_min'].mean() - 2*hc['putamen_min'].std()
        print(hc.shape)
        print(thresh)
        datscan['dat_deficit'] = datscan.apply(lambda row: row['putamen_min_age_corr'] < thresh, axis=1)
        datscan.loc[datscan['putamen_min'].isna(),'dat_deficit'] = np.nan
        return datscan
    else:
        datscan['age_group'] = pd.cut(datscan['visit_age'], bins=np.arange((datscan['visit_age'].min()//10)*10, (datscan['visit_age'].max()//10)*10+20, 10))
        hc = datscan[datscan['diagnosis']=='hc'].dropna(subset=['putamen_min'])
        # get group and age average
        thresh = hc['putamen_min'].mean() - 2*hc['putamen_min'].std()
        #thresh = hc['putamen_min'].quantile(.35)
        print(hc.shape)
        print(thresh)
        datscan['dat_deficit'] = datscan.apply(lambda row: row['putamen_min'] < thresh, axis=1).astype(int)
        datscan.loc[datscan['putamen_min'].isna(),'dat_deficit'] = np.nan
        return datscan

def get_ONOFF_med():
    med = pd.read_csv('/scratch/c.c21013066/data/ppmi/phenotypes2021/Use_of_PD_Medication.csv',usecols=['PATNO','INFODT','PDMEDYN','EVENT_ID'],
                     parse_dates=['INFODT'])
    # problem of duplicate date for V01 and V05, V05 should be three years later than shown
    med.loc[np.logical_and(med['PATNO']==4056,med['EVENT_ID']=='V05'),'INFODT'] += np.timedelta64(365,'D')
    med = med.drop(columns='EVENT_ID').set_index(['PATNO','INFODT'])
    med = med[~med.index.duplicated(keep='first')]
    med2 = pd.read_csv('/scratch/c.c21013066/data/ppmi/phenotypes/phenotypes/Laboratory_Procedures.csv',usecols=['PATNO','INFODT','PDMEDYN','EVENT_ID'],
                      parse_dates=['INFODT'],index_col=['PATNO','INFODT'])
    med2 = med2[~med2.index.duplicated(keep='first')]
    med3 = pd.read_csv('/scratch/c.c21013066/data/ppmi/phenotypes/phenotypes/MDS-UPDRS_Tagging_Status.csv',usecols=['PATNO','INFODT','ONOFFBN','EVENT_ID'],
                      parse_dates=['INFODT'],index_col=['PATNO','INFODT'])
    med3 = med3[~med3.index.duplicated(keep='first')]
    med3['PDMEDYN'] = (med3['ONOFFBN'] ==2).astype(int)
    med = reduce(lambda left,right: pd.merge(left,right,on=['PATNO','INFODT'],how='outer'),[med,med2,med3]).reset_index()
    med['ON'] = med[['PDMEDYN_x','PDMEDYN_y','PDMEDYN']].max(axis=1)
    med.rename(columns={'PATNO':'participant','INFODT':'date'},inplace=True)
    return med[['participant','date','ON']]

def get_n_visits(merged):
    n_visits = merged.groupby('participant').size().rename('n_visits')
    merged = pd.merge(merged,n_visits,on='participant',suffixes=['_drop',''])
    merged = merged.drop(columns=['n_visits_drop'])
    merged = merged[merged['n_visits']>=2]
    print("need at least 2 visits",merged.shape,len(np.unique(merged.index)))
    return merged