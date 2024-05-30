import numpy as np
import pandas as pd
import datetime
import calendar
from functools import reduce
import glob
import os

def age_seconds_to_years(df):
    return df['age_seconds']/31556952

def parse_dates(df):
    df['time_local'] = pd.to_datetime(df['time_local'])
    return df

def time_ms_to_date(df):
    return pd.DatetimeIndex(pd.to_datetime(df['time_ms'],unit='ms')).tz_localize('US/Pacific')

def get_date_from_age(df):
    return (df['date_birth'] + pd.to_timedelta(df['age_seconds'],unit='s').dt.floor('D'))

def get_date_local(df):
    
    return pd.to_datetime(df['date_start']) + pd.to_timedelta(df['time_local'].dt.hour,unit='h')

def ms_to_hours(df,measures):
    for measure in measures:
        df[f'{measure}_hour'] = df[f'{measure}_ms']/(1000*60*60)
    return df

def sleep_hours_to_percentage(df,measures):
    for measure in measures:
        df[f'{measure}_percent'] = (100/df[f'total_sleep_time_ms']) * df[f'{measure}_ms']
    return df

def align_weekday_date(df):
    df['date_local_adj']= adjust_date_fromweekday(df)
    return df

def adjust_date_fromweekday(day):
    weekdict = {'Mon':0,'Tue':1,'Wed':2,"Thu":3,'Fri':4,'Sat':5,'Sun':6}

    incorrectday = day['date_local'].dt.dayofweek
    correctday = day['time_day'].replace(weekdict)
    diff = correctday - incorrectday
    diff = diff.replace([6,-6,5,-5,4,-4],[-1,1,-2,2,-3,3])
    day['date_local_adj'] = day['date_local'] + pd.to_timedelta(diff,unit='d')
    return day['date_local_adj']
    
def process_raw(df_raw,demo):
    # merge with demo
    df_raw = df_raw.drop_duplicates(keep='first')
    df = pd.merge(df_raw,demo,right_on='participant',left_on='subject',how='left')
    df = parse_dates(df)
    # age to years
    df['age_accelerometry'] = age_seconds_to_years(df)
    df['time'] = time_ms_to_date(df)
    df['date'] = df['time'].dt.to_period('D')
    df['date_start'] = get_date_from_age(df) 
    df['date_local'] = get_date_local(df)
    df = align_weekday_date(df)
    # get time between visits
    df['time_between'] = df.groupby('subject')['date_local_adj'].diff()/ datetime.timedelta(minutes=1)    

    # derive features
    features = df.groupby('subject')[df_raw.columns[5:]].agg(['size','mean','std','max','min'])
    features.columns = features.columns.to_flat_index()
    return features, df

def clean_timeseries(df,keep='first'):
    df = df.dropna(subset=['date_local_adj'])
    return df.drop_duplicates(subset=['subject','date_local_adj'],keep='first')

def load_timeseries(demo,path='/scratch/c.c21013066/data/ppmi/accelerometer',names=[]):
    featuress = []
    ambulatory_raw = pd.read_csv(f'{path}/ambulatory.csv',na_values=['None'])
    features, ambulatory = process_raw(ambulatory_raw,demo)
    featuress.append(features)
    step_raw = pd.read_csv(f'{path}/stepcount.csv',na_values=['None'])
    features, step = process_raw(step_raw,demo)
    featuress.append(features)
    sleep_raw = pd.read_csv(f'{path}/sleepmetrics2.csv',na_values=['None'])
    measures = sleep_raw.iloc[:,5:].filter(regex='ms').columns.str.split('_ms').str[0]
    sleep_raw = ms_to_hours(sleep_raw,measures)
    features, sleep = process_raw(sleep_raw,demo)
    featuress.append(features)
    pulse_raw = pd.read_csv(f'{path}/pulserate.csv',na_values=['None'])
    features, pulse = process_raw(pulse_raw,demo)
    featuress.append(features)
    pulsevar_raw = pd.read_csv(f'{path}/prv.csv',na_values=['None'])
    features, pulsevar = process_raw(pulsevar_raw,demo)
    featuress.append(features)

    merged = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],
                                                how='outer'), featuress)
    return merged, ambulatory.set_index(['subject','date_local_adj']), step.set_index(['subject','date_local_adj']), sleep.set_index(['subject','date_local_adj']), pulse.set_index(['subject','date_local_adj']), pulsevar.set_index(['subject','date_local_adj'])

def load_specific_timeseries(demo,path='/scratch/c.c21013066/data/ppmi/accelerometer',names=[]):
    featuress = []
    raws = []
    for name in names:
        raw = pd.read_csv(f'{path}/{name}.csv',na_values=['None'])
        if name=='sleepmetrics2.csv':
            measures = sleep_raw.iloc[:,5:].filter(regex='ms').columns.str.split('_ms').str[0]
            raw = ms_to_hours(sleep_raw,measures)
        features, raw = process_raw(raw,demo)
        featuress.append(features)
        raws.append(raw)
    if len(names) > 1:
        merged = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],
                                                how='outer'), featuress)
        return merged, raws
    else:
        return featuress[0],raws[0]
    
def read_extracted_features(path,names=[]):
    if len(names)==0:
        names = [os.path.basename(x) for x in glob.glob(f'{path}/*.csv')]
    dfs = []
    for name in names:
        df = pd.read_csv(f'{path}/{name}')
        df['Unnamed: 0'] = df['Unnamed: 0'].astype(int)
        df = df.rename(columns={'Unnamed: 0':'participant'})
        dfs.append(df)
    return reduce(lambda  left,right: pd.merge(left,right,on='participant',
                                                how='outer'), dfs)
        

def merge_timeseries(dfs, how='outer',subset=[3086]):
    dfs = [df.loc[(subset,slice(None)),:] for df in dfs]
    merged = reduce(lambda  left,right: pd.merge(left,right,right_index=True,left_index=True,
                                                how=how,suffixes=['_drop','']), dfs)
    merged = merged.drop(columns=merged.filter(regex='_drop'))
    return merged

def merge_with_behavior(timeseries,behavior,tolerance=pd.Timedelta(60,'days'),on='behavior'):
    behavior = behavior.set_index('date').sort_index().reset_index()
    timeseries = timeseries.reset_index()
    timeseries = clean_timeseries(timeseries)
    timeseries = timeseries.drop(columns=['participant']).rename(columns={'subject':'participant'})
    timeseries = timeseries.set_index('date_local_adj').sort_index().reset_index()
    if on == 'behavior':
        return pd.merge_asof(behavior,timeseries,by='participant',right_on='date_local_adj',left_on='date', direction='nearest',tolerance=tolerance)
    else:
        return pd.merge_asof(timeseries,behavior,by='participant',left_on='date_local_adj',right_on='date', direction='nearest',tolerance=tolerance)