# -*- coding: utf-8 -*-
import codecs
import json
import os
from logging import getLogger, basicConfig

import numpy as np
import sys
import re


from datalib import config, DbaDumpLoader

basicConfig(level='INFO', format='%(asctime)s [%(levelname)s]: %(message)s')
force_build = len(sys.argv) > 1 and sys.argv[1] in ('-f', '--force')


def make_dba(table_id, data_id, suffix):
    #data_dir = os.path.abspath(os.path.join(config['DATA_ROOT'], 'dba_second/%s' % table_id))
    data_dir = os.path.abspath(os.path.join(config['DATA_ROOT'], 'purify/%s' % table_id))
    path_join = lambda *s: os.path.join(data_dir, *s)
    save_hdf = lambda d, p: d.to_hdf(p, '/data', mode='w', format='table')
    os.makedirs(data_dir, exist_ok=True)

    # check the output file
    if not force_build and os.path.exists(path_join('%s.hdf' % data_id)):
        return
    getLogger(__name__).info('[kpi "%s, %s"]: start to compile ...', table_id, data_id)

    # load and save the raw data
    loader = DbaDumpLoader(table_id, data_id, suffix=suffix, fillna=True,
                           purify=True, log_info=True)
    df = loader.load()
    save_hdf(df, path_join('%s.hdf' % data_id))

    # compute and save the standardized data
    non_missing_values = df[df['missing'] == 0]['value'].values
    mean, stddev = np.average(non_missing_values), np.std(non_missing_values)
    # with codecs.open(path_join('%s.std.json' % data_id), 'wb', 'utf-8') as f:
    #     f.write(json.dumps({'mean': mean, 'stddev': stddev, 'table_id': table_id, 'data_id': data_id}))
    if stddev == 0.0:
        getLogger(__name__).warning(
            '[kpi "%s, %s"]: zero derivation, data cannot be standardized.',
            table_id, data_id
        )
    else:
        df.loc[:,'value'] = (df['value'] - mean) / stddev
        df.loc[df['missing'] == 1, 'value'] = 0
        # save_hdf(df, path_join('%s.std.hdf' % data_id))


# for kpi in KPI_CONFIG:
#     Is_Dba = re.match(r'^monitor_label_data_', kpi.table)
#     if Is_Dba:
#         make_dba(kpi.table, kpi.key)

# for kpi in NEW_KPI_CONFIG:
#     make_dba(kpi.table, kpi.key)


def main(path):
    config.update(DATA_ROOT=path)
    folder = 'row_data'
    s = 'csv'
    KPI_list = []

    # for file in os.listdir(os.path.join(config['DATA_ROOT'], folder)):
    #     name = file.split('.')[0]
    #     suffix = file.split('.')[-1]
    #     if suffix == s:
    #         KPI_list.append(name)

    for file in os.listdir(os.path.join(config['DATA_ROOT'], folder)):
        suffix = file.split('.')[-1]
        if suffix == s:
            name = file.split('.%s' % s)[0]
            KPI_list.append(name)


    for key in KPI_list:
        make_dba(folder, key, s)


if __name__ == '__main__':
    main(sys.argv[1])

