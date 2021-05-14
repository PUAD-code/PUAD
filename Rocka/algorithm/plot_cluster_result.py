import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calendar import timegm
from datetime import datetime
from logging import getLogger
import matplotlib.lines as mlines
import os


DATA_PATH = '/home/jialingxiang/NewDTWFrame/Data'


def get_timestamp_for(year, month, day, hour=0, minute=0, second=0):
    """Get the correct timestamp for specified datetime literal."""
    dt = datetime(year, month, day, hour=hour, minute=minute, second=second)
    return timegm(dt.utctimetuple())


window_size = 120


# smooth the top 5% abnormal data.(in the whole curve)
def smoothing_method2(df):

    # z-normalization for the one-month data
    if 'missing' in df.columns:
        df.loc[df['missing'] == 1, 'value'] = np.nan
    df['value'].interpolate(inplace=True)
    filled_values = df['value'].values
    mean, stddev = np.average(filled_values), np.std(filled_values)
    df.loc[:,'value'] = (df['value'] - mean) / stddev

    df = df.reindex(df.value.abs().sort_values(inplace=False).index)
    most_deviate = int(len(df)*0.05)
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
        getLogger(__name__).info('kpi should be discarded due to std = 0')

    if df['value'].isnull().sum() > 0:
        getLogger(__name__).info('kpi should be discarded')
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
        getLogger(__name__).info('kpi should be discarded due to std(means) = 0')
    df = df[-len(means):]
    df['value'] = means

    return df


def standardization(df):
    df.loc[df['missing'] == 1, 'value'] = np.nan
    df['value'].interpolate(inplace=True)
    filled_values = df['value'].values
    mean, stddev = np.average(filled_values), np.std(filled_values)
    df.loc[:,'value'] = (df['value'] - mean) / stddev
    return df


df1 = pd.read_hdf(os.path.join(DATA_PATH, 'results_smooth_95_mean_std/cluster_result_r0.050000.hdf'))
df2 = pd.read_hdf(os.path.join(DATA_PATH, 'results_smooth_95_mean_std/classify_result_r0.050000.hdf'))

df = pd.concat([df1, df2], ignore_index=True)

cluster = {}

for item in df['uuid'].values:
    name = item
    cla = df[df['uuid']==item]['cluster'].values[0]
    if cla not in cluster.keys():
        cluster[cla] = [name]
    else:
        cluster[cla].append(name)

df = pd.read_hdf(os.path.join(DATA_PATH, 'results_smooth_95_mean_std/medoids_r0.050000.hdf'))
medoids = {}
for item in df['cluster'].values:
    medoid = df[df['cluster']==item]['medoid'].values[0]
    if item not in medoids.keys():
        medoids[item] = medoid
    else:
        raise ValueError('Multiple medoids.')


# start_ts = get_timestamp_for(2017, 10, 23, 0, 0, 0)
# end_ts = get_timestamp_for(2017, 10, 25, 0, 0, 0)
# start_ts = 1490707860 + 1440*60
# end_ts = 1493299800 - 1440*60
start_ts = 1490707860 + 1440*60*16
end_ts = 1490707860 + 1440*60*18


for i in cluster.keys():
    if i != -1:
        fig, ax = plt.subplots(1, 1)
        ts_copy = []
        for kpi in cluster[i]:
            df = pd.read_hdf(os.path.join(DATA_PATH, 'purify/raw_data/%s.hdf' % kpi))

            start_df = df[df['timestamp'] >= start_ts]
            df = start_df[start_df['timestamp'] < end_ts]

            df = standardization(df)

            value = df['value'].values
            ts = df['timestamp'].values
            ts_copy = ts

            ax.plot(ts, value, color='blue', linewidth=1, alpha=0.2)
        medoid = medoids[i]

        # plot centroid baseline
        df = pd.read_hdf(os.path.join(DATA_PATH, 'purify/cluster_data_smooth_95_mean/%s.std.hdf' % medoid))
        start_df = df[df['timestamp'] >= start_ts+60*60]
        end_df = start_df[start_df['timestamp'] < end_ts+60*60]
        df = end_df
        value = df['value'].values
        ax.plot(ts_copy, value, color='red', linewidth=1.5)
        ax.set_xticks([])
        ax.yaxis.set_tick_params(labelsize=16)

        # plot raw centroid
        # df = pd.read_hdf(os.path.join(DATA_PATH, 'purify/raw_data/%s.hdf' % medoid))
        # start_df = df[df['timestamp'] >= start_ts]
        # end_df = start_df[start_df['timestamp'] < end_ts]
        # df = standardization(end_df)
        # value = df['value'].values
        # ax.plot(ts_copy, value, color='red', linewidth=1)
        # ax.set_xticks([])
        # ax.yaxis.set_tick_params(labelsize=16)

        l1 = mlines.Line2D([], [], color='blue', linewidth=1, alpha=0.2, label='standardized KPIs')
        l2 = mlines.Line2D([], [], color='red', linewidth=1.5, label='cluster centroid baseline')

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.tight_layout()
        plt.figlegend([l1, l2], ['standardized KPIs', 'cluster centroid baseline'], loc='lower center', ncol=2,
                      bbox_to_anchor=(0.56, 1.0),
                      fontsize=16)

        plt.xticks(fontsize=14)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(DATA_PATH, 'results_smooth_95_mean_std/show_cluster%d.pdf' % i), bbox_inches='tight')


