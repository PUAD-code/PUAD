# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
from multiprocessing import Pool
from logging import getLogger

from datalib.loader import load_data_frame
from datalib import config

__all__ = ['get_all_kpis','form_training_matrix', 'get_data_array', 'split_dba_data']


# exp root and data path
EXP_ROOT = 'purify/cluster_data_smooth_95_mean/'
# EXP_ROOT = 'purify/cluster_data_without_smooth/'
# DATA_PATH = 'purify/goc/'
DATA_PATH = 'purify/row_data/'
os.makedirs(os.path.join(config['DATA_ROOT'], EXP_ROOT), exist_ok=True)


TOTAL_KPI = []
TRAIN_KPI = []
TEST_KPI = []

# test
START_TIME = config['START_TS']
END_TIME = config['END_TS']


window_size = config['WINDOW_SIZE']
# window_size = 1


def get_all_kpis():
    for file in os.listdir(os.path.join(config['DATA_ROOT'], DATA_PATH)):
        suffix = file.split('.')[-1]
        if suffix == 'hdf':
            name = file.split('.%s' % suffix)[0]
            TOTAL_KPI.append(name)
    arr = np.arange(0,len(TOTAL_KPI))
    np.random.shuffle(arr)
    bound = int(len(arr)*config['SAMPLE_RATE'])
    train = arr[:bound]
    if bound < len(arr):
        test = arr[bound:]
    else:
        test = []
    for i in train:
        TRAIN_KPI.append(TOTAL_KPI[i])
    for j in test:
        TEST_KPI.append(TOTAL_KPI[j])
    # for i in range(len(TOTAL_KPI)):
    #     if i % 9 == 0:
    #         TRAIN_KPI.append(TOTAL_KPI[i])
    #     else:
    #         TEST_KPI.append(TOTAL_KPI[i])
        # if i % 3 == 0:
        #     TEST_KPI.append(TOTAL_KPI[i])
        # else:
        #     TRAIN_KPI.append(TOTAL_KPI[i])
        # if int(TOTAL_KPI[i]) > 100:
        #     TEST_KPI.append(TOTAL_KPI[i])
        # else:
        #     TRAIN_KPI.append(TOTAL_KPI[i])
    return TOTAL_KPI, TRAIN_KPI, TEST_KPI


# make each data <= mean+k*std. Remove the anomaly whose value is too large, so that the cluster can be more accurate.
def purify_extreme_data(df):
    values = df['value'].values
    mean, std = np.average(values), np.std(values)
    # purify in windows:
    df.loc[np.fabs(df['value'] - mean) > 3*std, 'value'] = mean + 3*std
    # purify in the whole curve:
    #df.loc[np.fabs(df['value']) > 5, 'value'] = 5.0
    return df


def split_to_windows(df):
    buf = []
    last_pos = 0
    while last_pos + window_size <= len(df.index):
        chunk = df.iloc[last_pos:last_pos+window_size]
        pur_chunk = purify_extreme_data(chunk)
        buf.append(pur_chunk)
        last_pos += window_size
    chunk = df.iloc[last_pos:]
    pur_chunk = purify_extreme_data(chunk)
    buf.append(pur_chunk)

    if len(buf) == 1:
        new_df = buf[0]
    else:
        new_df = pd.concat(buf, ignore_index=True)

    return new_df


# only interpolate the missing and standardize data.
def smoothing_method1(paras):
    kpi, start, end = paras[0], paras[1], paras[2]

    df_read = load_data_frame(DATA_PATH + '%s.hdf' % kpi)
    # END_TIME = df_read['timestamp'].values[-1]
    # START_TIME = END_TIME - 43200 * 60
    df_start = df_read[df_read['timestamp'] >= START_TIME]
    df = df_start[df_start['timestamp'] < END_TIME]
    # z-normalization for the one-month data
    if 'missing' in df.columns:
        df.loc[df['missing'] == 1, 'value'] = np.nan
    df['value'].interpolate(inplace=True)
    filled_values = df['value'].values
    mean, stddev = np.average(filled_values), np.std(filled_values)
    if stddev != 0:
        df.loc[:,'value'] = (df['value'] - mean) / stddev
    else:
        getLogger(__name__).info('kpi %s should be discarded due to std = 0' % kpi)

    print('%s %d finished.' % (kpi, len(df)))
    df.to_hdf(os.path.join(config['DATA_ROOT'], EXP_ROOT + '%s.std.hdf' % kpi), '/cluster_data', mode='w',
              format='table')
    return kpi


# smooth the top 5% abnormal data.(in the whole curve)
def smoothing_method2(paras):
    kpi, start, end = paras[0], paras[1], paras[2]

    df_read = load_data_frame(DATA_PATH + '%s.hdf' % kpi)
    # END_TIME = df_read['timestamp'].values[-1]
    # START_TIME = END_TIME - 43200 * 60
    df_start = df_read[df_read['timestamp'] >= START_TIME]
    df = df_start[df_start['timestamp'] < END_TIME]
    # z-normalization for the one-month data
    if 'missing' in df.columns:
        df.loc[df['missing'] == 1, 'value'] = np.nan
    df['value'].interpolate(inplace=True)
    filled_values = df['value'].values
    mean, stddev = np.average(filled_values), np.std(filled_values)
    df.loc[:,'value'] = (df['value'] - mean) / stddev

    df = df.reindex(df.value.abs().sort_values(inplace=False).index)
    most_deviate = int(len(df)*0.1)
    val = df['value'].values
    val[-most_deviate:] = np.nan
    df['value'] = val
    df.sort_values('timestamp', inplace=True)
    df['value'].interpolate(inplace=True)
    if df['value'].isnull().sum() > 0:
        df['value'].fillna(method='ffill', inplace=True)
        df['value'].fillna(method='bfill', inplace=True)
    new_values = df['value'].values
    mean, std = np.mean(new_values), np.std(new_values)
    if std != 0:
        df.loc[:,'value'] = (df['value'] - mean) / std
    else:
        getLogger(__name__).info('kpi %s should be discarded due to std = 0' % kpi)

    if df['value'].isnull().sum() > 0:
        getLogger(__name__).info('kpi %s should be discarded' % kpi)
        df['value'].fillna(1, inplace=True)

    # get the mean curve
    values = df['value'].values
    means = []
    for i in range(window_size,len(values)):
        mean = np.mean(values[i-window_size:i])
        means.append(mean)
    if np.std(means)!=0:
        means = (means-np.mean(means))/np.std(means)
    else:
        getLogger(__name__).info('kpi %s should be discarded due to std(means) = 0' % kpi)
    df = df[-len(means):]
    df.loc[:,'value'] = means

    print('%s %d finished.' % (kpi, len(df)))
    df.to_hdf(os.path.join(config['DATA_ROOT'], EXP_ROOT + '%s.std.hdf' % kpi), '/cluster_data', mode='w',
                  format='table')
    return kpi


# split and store one-month data(dba). Without purify extreme data.
def split_dba_data(kpi_list, start, end):
    paras = [(kpi, start, end) for kpi in kpi_list]
    pool = Pool(20)
    result = pool.map(smoothing_method2, paras)
    # result = pool.map(smoothing_method1, paras)
    '''
    for kpi in kpi_list:
        df_read = load_data_frame(DATA_PATH + '%s.hdf' % kpi)
        df_start = df_read[df_read['timestamp'] >= START_TIME]
        df = df_start[df_start['timestamp'] < END_TIME]
        # z-normalization for the one-month data
        df.loc[df['missing'] == 1, 'value'] = np.nan
        df['value'].interpolate(inplace=True)
        filled_values = df['value'].values
        mean, stddev = np.average(filled_values), np.std(filled_values)
        df['value'] = (df['value'] - mean) / stddev

        # purify extreme data(optional)
        # new_df = split_to_windows(df)
        # new_df.loc[new_df['missing'] == 1, 'value'] = np.nan
        # new_df['value'].interpolate(inplace=True)
        # new_filled_values = new_df['value'].values
        # new_mean, new_stddev = np.average(new_filled_values), np.std(new_filled_values)
        # new_df['value'] = (new_df['value'] - new_mean) / new_stddev

        print('%s %d finished.' %(kpi, len(df)))
        df.to_hdf(os.path.join(config['DATA_ROOT'], EXP_ROOT + '%s.std.hdf' % kpi), '/cluster_data', mode='w', format='table')
    '''


# read the data frames. form the instances of the same kpi to a matrix.
def form_training_matrix(id_list):
    data_dict = {}
    for uuid in id_list:
        df = load_data_frame(EXP_ROOT + '%s.std.hdf' % str(uuid))
        data_dict[uuid] = df['value'].values
    return data_dict


def get_data_array(uuid):
    df = load_data_frame(EXP_ROOT + '%s.std.hdf' % str(uuid))
    data_array = df['value'].values
    return data_array


# fill the kpi_list. split and store one-month data for cluster.
# get_all_kpis()
# split_dba_data(TOTAL_KPI, START_TIME, END_TIME)
# data = form_training_matrix(TRAIN_KPI)
# print(data)
