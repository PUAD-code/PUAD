# -*- coding: utf-8 -*-
"""Methods and classes to load KPI data in DBA."""

import os
from itertools import chain

import numpy as np
import pandas as pd
from logging import getLogger

from . import config

__all__ = ['load_data_frame', 'DbaDumpLoader']


def load_data_frame(file_path, columns=None):
    """Load data frame from the data directory.

    Parameters
    ----------
    file_path : str
        Relative path of the data file name.

    columns : collections.Iterable[str]
        If specified, only these columns will be fetched.

    Returns
    -------
    pd.DataFrame
    """
    path = os.path.join(config['DATA_ROOT'], file_path)
    columns = list(columns) if columns else None
    suffix = path.split('.')[-1]
    if suffix == 'hdf':
        df = pd.read_hdf(path, columns=columns)
    elif suffix == 'csv':
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError('%s: invalid suffix. Need .csv or .hdf' % path)
    df.sort_values('timestamp', inplace=True)
    if 'id' in df.columns and (not columns or 'id' not in columns):
        df.drop('id', inplace=True)
    df.reset_index(inplace=True, drop=True)
    if 'label' not in df.columns:
        df['label'] = 0
    return df


def deduplicate_timestamp(df, table_id, data_id, log_info):
    ts = df['timestamp'].values
    pos = np.where(ts[1:] == ts[:-1])[0]
    if len(pos) > 0:
        pos = sorted(set(chain(pos, pos + np.asarray(1))))
        if log_info:
            getLogger(__name__).info(
                '%s, %s: %d points have duplicated timestamp.', table_id, data_id, len(pos))
        df2 = df.drop(df.index[pos]).reset_index()
        assert(len(df2) == len(df) - len(pos))
        ts2 = df2['timestamp'].values
        assert(len(np.where(ts2[1:] == ts2[:-1])[0]) == 0)
        df = df2
    return df


def purify_data_frame(df, table_id, data_id, fillna=True, log_info=True):
    df = df.reset_index(drop=True)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df = deduplicate_timestamp(df, table_id, data_id, log_info)
    ts = df['timestamp'].values

    if np.any(np.isnan(ts)):
        raise ValueError('%s, %s: some timestamp is NaN.' % (table_id, data_id))

    # aggregate the data to 1 min
    time_series = pd.Series(data=np.array(df['value']), index=pd.to_datetime(df['timestamp'], unit='s'))
    sampling_frequency = '5T'
    # sampling_frequency = '10s'
    sampled_series = time_series.resample(sampling_frequency, closed='left', label='left').mean()
    label_series = pd.Series(data=np.array(df['label']), index=pd.to_datetime(df['timestamp'], unit='s'))

    #label=1 as long as there is an anomaly label in this minute.
    sampled_label = label_series.resample(sampling_frequency, closed='left', label='left').max()
    sampled_df = pd.DataFrame(sampled_series.index, columns=['timestamp'])
    sampled_df['value'] = sampled_series.values
    sampled_df['label'] = sampled_label.values
    #timestamp: s
    sampled_df['timestamp'] = sampled_df['timestamp'].values.astype(np.int64) // 10**9

    interval = 300    #interval 60s
    # interval = 10

    #split the data frame into continuous chunks, and fill the gap
    def fill_break(start_ts, end_ts):
        assert((end_ts - start_ts) % interval == 0)
        gap_size = (end_ts - start_ts) // interval
        col_ts = np.arange(start_ts, end_ts, interval)
        col_ts = col_ts.astype(sampled_df['timestamp'].dtype)
        fill = {'timestamp': col_ts}
        for k in sampled_df:
            if k != 'timestamp':
                if k == 'value':
                    fill[k] = np.full(gap_size, np.nan, dtype=sampled_df[k].dtype)
                else:
                    fill[k] = np.zeros(gap_size, dtype=sampled_df[k].dtype)
        return pd.DataFrame.from_dict(fill)

    #find breaks
    sampled_ts = sampled_df['timestamp'].values
    buf = []
    breaks = np.where(sampled_ts[1:] - sampled_ts[:-1] > interval)[0] + np.asarray(1)
    last_pos = 0
    nan_count = 0

    #fill the breaks
    for b in breaks:
        buf.append(sampled_df.iloc[last_pos:b])
        fill = fill_break(sampled_df['timestamp'].iloc[b-1] + interval, sampled_df['timestamp'].iloc[b])
        buf.append(fill)
        nan_count += len(fill)
        last_pos = b

    buf.append(sampled_df.iloc[last_pos:])

    #concatenate the chunks
    if len(buf) == 1:
        sampled_df = buf[0]
    else:
        sampled_df = pd.concat(buf, ignore_index=True)

    #check the integrity of concatenated data frame
    sampled_ts2 = sampled_df['timestamp'].values
    assert(sampled_df.index[-1] == len(sampled_df) - 1)
    assert(sampled_ts2[0] == sampled_ts[0])
    assert(sampled_ts2[-1] == sampled_ts[-1])
    assert(len(np.unique(sampled_ts2[1:] - sampled_ts2[:-1])) == 1)
    assert(sampled_ts2[1] - sampled_ts2[0] == interval)

    #add a column to indicate missing values, and fill the missing values with zeros.
    if fillna:
        sampled_df['missing'] = pd.isnull(sampled_df['value']).astype(np.int32)
        assert (np.sum(sampled_df['missing']) == np.sum(np.isnan(sampled_df['value'].values)))
        sampled_df['value'] = sampled_df['value'].interpolate(method='linear')
        sampled_df['label'] = sampled_df['label'].interpolate(method='linear')
        sampled_df['label'] = np.array([round(x) for x in sampled_df['label']])
        # sampled_df['value'] = sampled_df['value'].fillna(0.)
        # sampled_df['label'] = sampled_df['label'].fillna(0).astype(np.int32)

    return sampled_df


'''
df = load_data_frame('dump/monitor_label_data_0000/kpi_001.hdf')
df1 = purify_data_frame(df, 'monitor_label_data_0000', 'kpi_001')
print(df1)
'''

class DbaDumpLoader(object):
    """Class to load DBA dumped data.

    Parameters
    ----------
    table_id : str
        table name of the KPI data in the database.

    data_id : str
        data id of the KPI data

    fillna : bool
        Whether or not to fill NaN values column?

        If True, will set all NaN values in `value` column to zero,
        and attach an extra `missing` column to indicate whether
        or not the `value` column is NaN for each row.

        If False, will not set NaN values to zero, neither will the
        `missing` column be attached.

        Note that the `label` column will always be filled with zero
        for NaN, and the data type of `label` column will be changed
        to int32.

    purify : bool
        Whether or not to purify the data?

        Purified data will be ensured to have continuous and ordered timestamps
        in homogeneous time intervals.

    log_info : bool
        Whether or not to log information of the data?
    """

    def __init__(self, table_id, data_id, suffix='hdf', fillna=True, purify=True,
                 log_info=True):
        self.table_id = table_id
        self.data_id = data_id
        self.suffix = suffix
        self.fillna = fillna
        self.purify = purify
        self.log_info = log_info

    def __repr__(self):
        return 'DbaLocalLoader(%r, %r)' % (self.table_id, self.data_id,)

    def load(self):
        """Load purified KPI data from dumped file."""
        df = load_data_frame('%s/%s.%s' % (self.table_id, self.data_id, self.suffix))

        # fill NaN of label column and change data type
        df['label'] = df['label'].fillna(0).astype(np.int32)

        # purify the loaded data
        if self.purify:
            df = purify_data_frame(
                df, '[kpi "%s"]' % self.table_id, '%s' % self.data_id, fillna=self.fillna,
                log_info=self.log_info
            )

        if self.log_info:
            label_count = df['label'].sum()
            if 'missing' in df:
                missing_count = df['missing'].sum()
            else:
                missing_count = df['value'].isnull().sum()

            getLogger(__name__).info(
                '[kpi "%s %s"]: finished, %s (%.2f%%) labels, '
                '%s (%.2f%%) missing.',
                self.table_id,
                self.data_id,
                label_count,
                label_count * 100. / len(df),
                missing_count,
                missing_count * 100. / len(df)
            )

        return df
