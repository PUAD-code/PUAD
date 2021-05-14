# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import codecs
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import time
from logging import getLogger, basicConfig
from datetime import datetime
from multiprocessing import Pool
import matplotlib.lines as mlines

from explib.cluster.data_prepare import form_training_matrix, get_data_array, get_all_kpis, split_dba_data, START_TIME, END_TIME
from explib.cluster.distance_measure import SBD, fast_dtw, l1_norm, Euclidean, SBD_no_shift
from datalib import config

from scipy.spatial.distance import euclidean

pd.options.mode.chained_assignment = None

__all__ = ['Clusterer', 'Evaluator', 'draw_cluster_medoids', 'draw_each_cluster']

# dba exp root
# EXP_ROOT = '/mnt/mfs/users/lzh/data/ali/ver2/results/'
EXP_ROOT = os.path.join(config['DATA_ROOT'], 'results_smooth_95_mean_std/')
# EXP_ROOT = os.path.join(config['DATA_ROOT'], 'results_95_baseline_DTW/')
os.makedirs(EXP_ROOT, exist_ok=True)

basicConfig(level='INFO', format='%(asctime)s [%(levelname)s]: %(message)s',
            filename=EXP_ROOT + 'exp_log.log', filemode='w')

TOTAL_KPI, TRAIN_KPI, TEST_KPI = get_all_kpis()
split_dba_data(TOTAL_KPI, START_TIME, END_TIME)

# get the training data
data_dict = form_training_matrix(TRAIN_KPI)

all_kpi = []
all_cla = []
all_dist = []



def get_similarity(paras):
    i, j, dist = paras[0],paras[1],paras[2]
    if dist == 'SBD':
        sim, y, shift = SBD(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
    elif dist == 'SBD_no_shift':
        sim, y, shift = SBD_no_shift(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
    elif dist == 'DTW':
        sim = fast_dtw(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]], dist_measure=euclidean)
        shift = 0
    elif dist == 'L1':
        sim = l1_norm(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
        shift = 0
    elif dist == 'Euclidean':
        sim = Euclidean(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
        shift = 0
    else:
        raise ValueError("Unexpected parameter value dist=%s." % dist)
    print('sim between %s and %s is %f' % (TRAIN_KPI[i], TRAIN_KPI[j], sim))
    return tuple([TRAIN_KPI[i], TRAIN_KPI[j], sim, shift])


def training_similarity_matrix_dba(dist):
    '''
    calculate similarity matrix of the training data.
    :param dist: str.
    The distance measure to measure similarity between time series. It can be 'SBD', 'DTW', 'L1', 'Euclidean'.
     Default 'SBD'.
    :return:
    '''
    print(len(data_dict))
    start = time.time()
    sim_matrix = np.zeros(shape=(len(data_dict), len(data_dict))) - 1
    buf = []
    paras = []
    for i in range(len(TRAIN_KPI)):
        for j in range(len(TRAIN_KPI)):
            if i < j:
                paras.append((i,j,dist))
    pool = Pool(20)
    similarities = pool.map(get_similarity, paras)
    end = time.time()
    getLogger(__name__).info('training similarity matrix time: %f' % (end - start))
    for item in similarities:
        buf.append(tuple([item[1],item[0],item[2],-item[3]]))
    similarities.extend(buf)
    result = pd.DataFrame([x for x in similarities], columns=['start', 'end', 'sim', 'shift'])
        # i, j, sim, shift = item[0], item[1], item[2], item[3]
        # sim_matrix[i][j] = sim
        # sim_matrix[j][i] = sim_matrix[i][j]
        # fill = {'start': [TRAIN_KPI[i], TRAIN_KPI[j]], 'end': [TRAIN_KPI[j], TRAIN_KPI[i]],
        #         'sim': [sim_matrix[i][j], sim_matrix[j][i]], 'shift': [shift, -shift]}
        # df = pd.DataFrame.from_dict(fill)
        # # print(df)
        # buf.append(df)
    # for i in range(len(TRAIN_KPI)):
    #     for j in range(len(TRAIN_KPI)):
    #         if i < j:
    #             if dist == 'SBD':
    #                 sim_matrix[i][j], y, shift = SBD(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
    #             elif dist == 'DTW':
    #                 sim_matrix[i][j] = fast_dtw(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]], dist_measure=euclidean)
    #                 shift = 0
    #             elif dist == 'L1':
    #                 sim_matrix[i][j] = l1_norm(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
    #                 shift = 0
    #             elif dist == 'Euclidean':
    #                 sim_matrix[i][j] = Euclidean(data_dict[TRAIN_KPI[i]], data_dict[TRAIN_KPI[j]])
    #                 shift = 0
    #             else:
    #                 raise ValueError("Unexpected parameter value dist=%s." % dist)
    #             sim_matrix[j][i] = sim_matrix[i][j]
    #             print('sim between %s and %s is %f' % (TRAIN_KPI[i], TRAIN_KPI[j], sim_matrix[i][j]))
    #             fill = {'start': [TRAIN_KPI[i], TRAIN_KPI[j]], 'end':[TRAIN_KPI[j], TRAIN_KPI[i]],
    #                     'sim': [sim_matrix[i][j], sim_matrix[j][i]], 'shift': [shift, -shift]}
    #             df = pd.DataFrame.from_dict(fill)
    #             #print(df)
    #             buf.append(df)
    # print('training similarity matrix time: %f' % (end - start))
    # result = pd.concat(buf, ignore_index=True)
    # result = result[['start', 'end', 'shift', 'sim']]
    result.to_hdf(EXP_ROOT + 'sim_matrix.hdf', '/sim_matrix', mode='w', format='table')


def get_sim_matrix(df):
    matrix = np.zeros((len(TRAIN_KPI), len(TRAIN_KPI)))
    d = dict(zip(zip(df['start'].values, df['end'].values), df['sim'].values))
    for i in range(0, len(TRAIN_KPI)):
        for j in range(0, len(TRAIN_KPI)):
            if i != j:
                matrix[i][j] = d[(TRAIN_KPI[i], TRAIN_KPI[j])]
                # tmp = df[df['start'] == TRAIN_KPI[i]]
                # tmp1 = tmp[tmp['end'] == TRAIN_KPI[j]]
                # matrix[i][j] = tmp1['sim']
                # matrix[j][i] = matrix[i][j]
    return matrix


'''
recursively calculate the inflection points on the k-dis curve. start and end are the first and last index of the
selected part of the curve. In the region (start,end], if |slope(start,i) - slope(i,end)| < threshold, then i is regarded
as an inflection point, and its corresponding value is a radius.
'''
def inflection_point(radius, k_dis, start, end, threshold):
    r = -1
    '''
    k_dis is a sorted list, the curve is monotonically increasing. The max Y-value is smaller than two, so the diff
    between two parts of curve is also smaller than 2.
    '''
    diff = 2
    if end - start <= 15:
        return
    for i in range(start, end):
        if i == start:
            left = 0
        else:
            left = (k_dis[i] - k_dis[start]) / (i - start)
        right = (k_dis[end] - k_dis[i]) / (end - i)
        if left > 0.02 or right > 0.02:
            continue
        #print(abs(right - left), i)
        if abs(right - left) < diff:
            diff = abs(right - left)
            r = i

    #print(diff, r)
    if diff < threshold:
        radius.append(r)
    inflection_point(radius, k_dis, start, r-1, threshold)
    inflection_point(radius, k_dis, r+1, end, threshold)


# calculate 3-NN similarity distance of the training data and determine the density radius.
def density_estimation(df, max_radius, inflect_thresh, min_samples):
    three_dis = []
    for uuid in TRAIN_KPI:
        tmp = df[df['start'] == uuid]
        sim_list = tmp['sim'].values
        sorted_sim = np.sort(sim_list)
        three_dis.append(sorted_sim[min_samples-2])     # the third nearest neighbor of curve 'uuid'.
    three_dis.sort(reverse=True)
    sorted_three_dis = three_dis
    #print('result:', sorted_three_dis)

    length = len(sorted_three_dis)

    # all candidate density radius list
    radius = []

    start = time.time()
    inflection_point(radius, sorted_three_dis, 0, length-1, threshold=inflect_thresh)    #0.008
    # print(radius)
    ra_vals = []
    # large radius is meaningless because the distance measure already shows the similarity between curves.
    for ra in radius:
        # print(ra, sorted_three_dis[ra])
        if sorted_three_dis[ra] < max_radius:       #0.05
            ra_vals.append(sorted_three_dis[ra])
    end = time.time()
    getLogger(__name__).info('Density estimation time: %f' % (end-start))

    plt.figure(figsize=(4,2))
    x = range(0,length)
    y = sorted_three_dis
    plt.plot(x,y)
    # plt.plot(sorted_three_dis.index(max(ra_vals)), max(ra_vals), 'ro')
    plt.savefig(EXP_ROOT + 'sorted_three_dis.pdf')
    np.save(EXP_ROOT + 'sorted_k_dis.npy', sorted_three_dis)
    return ra_vals


# DBSCAN cluster algorithm with the calculated radius.
def dbscan(sim_matrix, radius, min_samples, dist_measure, max_radius):
    start = datetime.now()
    result = DBSCAN(eps=radius, min_samples=min_samples, metric='precomputed', n_jobs=-1).fit(sim_matrix)
    end = datetime.now()
    getLogger(__name__).info('dbscan running time:{}'.format(end-start))
    core_sample_mask = np.zeros_like(result.labels_, dtype=bool)
    core_sample_mask[result.core_sample_indices_] = True
    labels_cal = result.labels_

    # for i in range(0,len(result.labels_)):
    #     print(TRAIN_KPI[i], core_sample_mask[i], labels_cal[i])

    # Number of clusters in labels calculated by DBSCAN, ignoring noise if present
    num_clusters = len(set(labels_cal)) - (1 if -1 in labels_cal else 0)

    # print('number of clusters: %d' % num_clusters)
    getLogger(__name__).info('number of clusters: %d' % num_clusters)

    cluster = {}
    medoids = []

    for cla in range(0, num_clusters):
        cluster[cla] = []
        # print('class %d: %d' % (cla, labels_cal.tolist().count(cla)))
        getLogger(__name__).info('class %d: %d' % (cla, labels_cal.tolist().count(cla)))
        index = [idx for idx, e in enumerate(labels_cal) if e == cla]
        for id in index:
            cluster[cla].append(TRAIN_KPI[id])
            # print(TRAIN_KPI[id])

        medoid, min_dist = get_the_medoids(sim_matrix, index)
        medoids.append(medoid)
        # print(medoid, min_dist)
        getLogger(__name__).info(medoid)
        getLogger(__name__).info(min_dist)

    # assign the 'noisy' curve in DBSCAN and find the real noise.(sim to all the medoids are larger than threshold)
    index = [idx for idx, e in enumerate(labels_cal) if e == -1]
    cluster[-1] = []
    # assign according to the sim to the medoid of each cluster.
    for uuid in index:
        data_arr = data_dict[TRAIN_KPI[uuid]]
        cla,it_dist = assignment(medoids, data_arr, dist_category=dist_measure)
        cluster[cla].append(TRAIN_KPI[uuid])
        labels_cal[uuid] = cla
        print('KPI %s belongs to class %d' %(TRAIN_KPI[uuid], cla))
        getLogger(__name__).info('KPI %s belongs to class %d' %(TRAIN_KPI[uuid], cla))
        all_kpi.append(TRAIN_KPI[uuid])
        all_cla.append(cla)
        all_dist.append(it_dist)
    dataframe = pd.DataFrame({'uuid': all_kpi, 'cluster': all_cla, 'dist': all_dist})
    dataframe.to_csv("/home/jialingxiang/NewDTWFrame/SplitKPI/all_dist.csv", index=False, sep=',')


    # assign method 2: assign according to its nearest clustered curve.
    '''
    for uuid in index:
        cla, new_labels = assign_to_nearest(sim_matrix, uuid, labels_cal)
        cluster[cla].append(TRAIN_KPI[uuid])
        print('KPI %d belongs to class %d' %(TRAIN_KPI[uuid], cla))
        labels_cal = new_labels
    '''

    # print(cluster)
    result = {}
    for key in cluster.keys():
        # print('class %d: %d' % (key, len(cluster[key])))
        getLogger(__name__).info('class %d: %d' % (key, len(cluster[key])))
        for value in cluster[key]:
            result[value] = key
    result_df = pd.DataFrame(list(result.items()), columns=['uuid', 'cluster'])
    result_df.to_hdf(EXP_ROOT + 'cluster_result_r%f.hdf' % max_radius, '/cluster_result', mode='w',
                  format='table')

    medoids_dict = {}
    for i in range(0,len(medoids)):
        medoids_dict[i] = medoids[i]
    medoids_df = pd.DataFrame(list(medoids_dict.items()), columns=['cluster', 'medoid'])
    medoids_df.to_hdf(EXP_ROOT + 'medoids_r%f.hdf' % max_radius, '/medoids', mode='w',
                  format='table')


    return medoids, labels_cal


# assign the noise in dbscan to the cluster which its nearest neighbor belongs to.
def assign_to_nearest(sim_matrix, noise_index, labels):
    sim_dis = sim_matrix[noise_index][:]
    sim_dis_copy = sim_dis.tolist()
    sim_dis.sort()
    for i in sim_dis:
        if i < 0.05:
            index = sim_dis_copy.index(i)
            if labels[index] != -1:
                labels[noise_index] = labels[index]
                return labels[noise_index], labels
            else:
                continue
        else:
            labels[noise_index] = -1
            return labels[noise_index], labels
    return labels[noise_index], labels


# calculate the center curve of each cluster, whose sum(square(sim to other curves in the same cluster)) is the smallest.
def get_the_medoids(sim_matrix, indexes):
    min_dist = sys.maxsize
    medoid = -1
    for start in indexes:
        dist = 0
        for end in indexes:
            if end != start:
                dist += np.square(sim_matrix[start][end])
        if dist < min_dist:
            min_dist = dist
            medoid = start
    return TRAIN_KPI[medoid], min_dist


# assign the other curves to the closest cluster.
def assignment(medoids, arr, dist_category):
    # set a threshold. If the sim between a curve and all medoids are larger than threshold, then this curve is regarded as noise.
    min_dist = 0.2
    # min_dist = 0.4
    # if assign all time series to an existing cluster.
    # min_dist = 2
    cla = -1
    for i in range(0, len(medoids)):
        medoid_id = medoids[i]
        medoid_arr = data_dict[medoid_id]
        if dist_category == 'SBD':
            dist, y, shift = SBD(medoid_arr, arr)
        elif dist_category == 'SBD_no_shift':
            dist, y, shift = SBD_no_shift(medoid_arr, arr)
        elif dist_category == 'DTW':
            dist = fast_dtw(medoid_arr, arr, dist_measure=euclidean)
        elif dist_category == 'L1':
            dist = l1_norm(medoid_arr, arr)
        elif dist_category == 'Euclidean':
            dist = Euclidean(medoid_arr, arr)
        else:
            raise ValueError("Unexpected parameter value dist=%s." % dist_category)
        if dist < min_dist:
            min_dist = dist
            cla = i
    
    return cla,min_dist




def get_ground_truth(path, kpi_list):
    df = pd.read_csv(path)
    df['name'] = df['name'].astype(str)
    y_true = []
    for kpi in kpi_list:
        # print(kpi, df.dtypes)
        df1 = df[df['name'] == kpi]
        cla = df1['class'].values
        y_true.append(int(cla[0]))
    return y_true


# assign the unclustered curves.
def classify(index_list, medoids, dist, max_radius):
    test_pred = []
    classify_result = {}
    classify_result[-1] = []
    for i in range(0,len(medoids)):
        classify_result[i] = []

    for uuid in index_list:
        data_arr = get_data_array(uuid)
        print(uuid,end=" ")
        cla,it_dist = assignment(medoids, data_arr, dist_category=dist)
        all_kpi.append(uuid)
        all_cla.append(cla)
        all_dist.append(it_dist)
        test_pred.append(cla)
        print('KPI %s belongs to class %d' % (uuid, cla))
        getLogger(__name__).info('KPI %s belongs to class %d' % (uuid, cla))
        classify_result[cla].append(uuid)

    dataframe = pd.DataFrame({'uuid': all_kpi, 'cluster': all_cla, 'dist': all_dist})
    dataframe.to_csv("/home/jialingxiang/NewDTWFrame/all_dist.csv", index=False, sep=',')


    print(classify_result)
    # print(classify_result)
    getLogger(__name__).info(classify_result)
    result = {}
    for key in classify_result.keys():
        # print('class %d: %d' % (key, len(classify_result[key])))
        getLogger(__name__).info('class %d: %d' % (key, len(classify_result[key])))
        for value in classify_result[key]:
            result[value] = key
    result_df = pd.DataFrame(list(result.items()), columns=['uuid', 'cluster'])
    result_df.to_hdf(EXP_ROOT + 'classify_result_r%f.hdf' % max_radius, '/classify_result',
                     mode='w',format='table')

    return test_pred


def match_class_labels(y_true, y_pred):
    pred_dict = {}
    for i in np.arange(0,len(y_pred)):
        if y_pred[i] not in pred_dict.keys():
            pred_dict[y_pred[i]] = [i]
        else:
            pred_dict[y_pred[i]].append(i)
    true_classes = []
    for j in y_true:
        if j not in true_classes and j != -1:
            true_classes.append(j)
    true_class_num = len(true_classes)     # except the outlier class -1
    label_mappings = {}
    for cla in pred_dict.keys():
        if cla != -1:
            slot = np.zeros(true_class_num)
            for tmp in pred_dict[cla]:
                match_label = y_true[tmp]
                if match_label != -1:
                    slot[match_label] += 1
            real_cla = np.argmax(slot)
            label_mappings[cla] = real_cla
            for item in pred_dict[cla]:
                y_pred[item] = real_cla
    return y_pred, label_mappings


def match_test_labels(test_pred, label_mappings):
    for i in np.arange(0, len(test_pred)):
        if test_pred[i] != -1:
            real_label = label_mappings[test_pred[i]]
            test_pred[i] = real_label
    return test_pred


def evaluation_pr(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    fscore = metrics.f1_score(y_true, y_pred, average='macro')
    # print('Precision score: %f, Recall score: %f' % (precision, recall))
    getLogger(__name__).info('Precision score: %f, Recall score: %f' % (precision, recall))
    getLogger(__name__).info('NMI score: %f' % nmi)
    getLogger(__name__).info('F-score: %f' % fscore)


def evaluation_pr_ignore_outlier(y_true, y_pred):
    pred = []
    true = []
    for i in np.arange(0,len(y_pred)):
        if y_pred[i] != -1:
            pred.append(y_pred[i])
            true.append(y_true[i])
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(true, pred)
    fscore = metrics.f1_score(true, pred, average='macro')
    # print('Ignore outlier Precision score: %f, Recall score: %f' % (precision, recall))
    getLogger(__name__).info('Ignore outlier Precision score: %f, Recall score: %f' % (precision, recall))
    getLogger(__name__).info('NMI score: %f' % nmi)
    getLogger(__name__).info('F-score: %f' % fscore)


def draw_cluster_medoids(max_radius):
    df = pd.read_hdf(os.path.join(EXP_ROOT, 'medoids_r%f.hdf' % max_radius))
    num = len(df)
    if num <= 0:
        raise ValueError('No clusters found.')
    elif num == 1:
        fig, ax = plt.subplots()
        medoids = df['medoid'].values
        y = get_data_array(medoids[0])
        ax.plot(y)
    else:
        fig, axarr = plt.subplots(num, sharex=True, sharey=True)
        medoids = df['medoid'].values
        for i in range(num):
            medoid = medoids[i]
            y = get_data_array(medoid)
            axarr[i].plot(y)

    plt.savefig(os.path.join(EXP_ROOT, 'show_medoids_r%f.pdf' % max_radius))


def standardization(df):
    df.loc[df['missing'] == 1, 'value'] = np.nan
    df['value'].interpolate(inplace=True)
    filled_values = df['value'].values
    mean, stddev = np.average(filled_values), np.std(filled_values)
    df.loc[:,'value'] = (df['value'] - mean) / stddev
    return df


def draw_each_cluster(max_radius, start_ts, end_ts):
    df1 = pd.read_hdf(os.path.join(config['DATA_ROOT'], 'results_smooth_95_mean_std/cluster_result_r%f.hdf' % max_radius))
    if config['SAMPLE_RATE'] < 1:
        df2 = pd.read_hdf(os.path.join(config['DATA_ROOT'], 'results_smooth_95_mean_std/classify_result_r%f.hdf' % max_radius))
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    output_path = os.path.join(config['DATA_ROOT'], config['OUTPUT_PATH'])
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(os.path.join(output_path, 'result.csv'))

    cluster = {}

    for item in df['uuid'].values:
        name = item
        cla = df[df['uuid'] == item]['cluster'].values[0]
        if cla not in cluster.keys():
            cluster[cla] = [name]
        else:
            cluster[cla].append(name)

    df = pd.read_hdf(os.path.join(config['DATA_ROOT'], 'results_smooth_95_mean_std/medoids_r%f.hdf' % max_radius))
    medoids = {}
    for item in df['cluster'].values:
        medoid = df[df['cluster'] == item]['medoid'].values[0]
        if item not in medoids.keys():
            medoids[item] = medoid
        else:
            raise ValueError('Multiple medoids.')

    for i in cluster.keys():
        if i != -1:
            fig, ax = plt.subplots(1, 1)
            ts_copy = []
            for kpi in cluster[i]:
                df = pd.read_hdf(os.path.join(config['DATA_ROOT'], 'purify/row_data/%s.hdf' % kpi))

                start_df = df[df['timestamp'] >= start_ts]
                df = start_df[start_df['timestamp'] < end_ts]

                df = standardization(df)

                value = df['value'].values
                ts = df['timestamp'].values
                ts_copy = ts
                #print("ts is:",end="")
                #print(ts)
                #print("value.shape is:",end="")
                #print(value.shape)
                #print()
                ax.plot(ts, value, color='blue', linewidth=1, alpha=0.2)
            medoid = medoids[i]

            # plot centroid baseline
            df = pd.read_hdf(os.path.join(config['DATA_ROOT'], 'purify/cluster_data_smooth_95_mean/%s.std.hdf' % medoid))
            start_df = df[df['timestamp'] >= start_ts + 60 * 300]
            end_df = start_df[start_df['timestamp'] < end_ts + 60 * 300]
            df = end_df
            value = df['value'].values
            #print("value is:")
            #print(value)
            #print("ts_copy is:",end="")
            #print(ts_copy)
            #print("value.shape is:",end="")
            #print(value.shape)
            ax.plot(ts_copy, value, color='red', linewidth=1.5)
            ax.set_xticks([])
            ax.yaxis.set_tick_params(labelsize=16)

            l1 = mlines.Line2D([], [], color='blue', linewidth=1, alpha=0.2, label='standardized KPIs')
            l2 = mlines.Line2D([], [], color='red', linewidth=1.5, label='cluster centroid baseline')

            fig.subplots_adjust(hspace=0, wspace=0)
            plt.tight_layout()
            plt.figlegend([l1, l2], ['standardized KPIs', 'cluster centroid baseline'], loc='lower center', ncol=2,
                          bbox_to_anchor=(0.56, 1.0),
                          fontsize=16)

            plt.xticks(fontsize=14)
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(output_path, 'show_cluster%d.pdf' % i),
                        bbox_inches='tight')
            plt.savefig(os.path.join(output_path, 'show_cluster%d.png' % i),
                        bbox_inches='tight')


class Clusterer(object):

    def __init__(self, min_samples=5, dist_measure='SBD',
                 max_radius=0.05, inflect_thresh=0.005, train_sim_matrix=True):
        self.min_samples = min_samples
        self.dist_measure = dist_measure
        self.max_radius = max_radius
        self.inflect_thresh = inflect_thresh
        self.train_sim_matrix = train_sim_matrix

    def run(self):
        # calculate similarity matrix for training set. only need to do once for a specific training set.
        if self.train_sim_matrix:
            st = time.time()
            training_similarity_matrix_dba(self.dist_measure)
            en = time.time()
            # print('training similarity matrix calculation time:%f s.' % (en - st))
            getLogger(__name__).info('training similarity matrix calculation time:%f s.' % (en - st))

        # read the similarity dataframe of training data.
        sim_df = pd.read_hdf(os.path.join(EXP_ROOT, 'sim_matrix.hdf'))

        # cluster part
        st = time.time()
        # print("Cluster set KPI num:%d" % len(TRAIN_KPI))
        getLogger(__name__).info("Cluster set KPI num:%d" % len(TRAIN_KPI))
        ra_vals = density_estimation(sim_df, self.max_radius, self.inflect_thresh, self.min_samples)
        # ra_vals = [0.05]
        # print(ra_vals)
        getLogger(__name__).info(ra_vals)
        sim_matrix = get_sim_matrix(sim_df)
        # print(sim_matrix)
        # print('density cluster radius: %f' % max(ra_vals))
        getLogger(__name__).info('density cluster radius: %f' % max(ra_vals))
        medoids, y_pred = dbscan(sim_matrix, max(ra_vals), self.min_samples, self.dist_measure, self.max_radius)
        en = time.time()
        cluster_time = en - st
        # print('cluster time: %f s.' % cluster_time)
        getLogger(__name__).info('cluster time: %f s.' % cluster_time)

        # classify part
        st = time.time()
        # print(len(TEST_KPI))
        getLogger(__name__).info(len(TEST_KPI))
        # assign
        test_pred = classify(TEST_KPI, medoids, dist=self.dist_measure, max_radius=self.max_radius)
        en = time.time()
        if len(TEST_KPI)>0:
            avg_classify_time = (en - st) / len(TEST_KPI)
        else:
            avg_classify_time = -1
        # print('test time:%f s., avg test time:%f s.' % ((en - st), avg_classify_time))
        getLogger(__name__).info('test time:%f s., avg test time:%f s.' % ((en - st), avg_classify_time))
        # print('density cluster radius: %f' % max(ra_vals))
        getLogger(__name__).info('density cluster radius: %f' % max(ra_vals))

        with codecs.open(os.path.join(EXP_ROOT, 'results_r%f.json' % self.max_radius), 'wb', 'utf-8') as f:
            f.write(json.dumps({'min_samples': self.min_samples, 'dist_measure': self.dist_measure,
                                'max_radius': self.max_radius, 'inflect_thresh': self.inflect_thresh,
                                'cluster_radius': max(ra_vals), 'cluster_KPI_num': len(TRAIN_KPI),
                                'cluster_time': cluster_time, 'avg_classify_time': avg_classify_time}))

        return y_pred, test_pred

class Evaluator(object):

    def __init__(self, y_pred, test_pred, ignore_outlier=False):
        self.y_pred = y_pred
        self.test_pred = test_pred
        self.ignore_outlier = ignore_outlier

    def run(self):
        # cluster result evaluation
        y_true = get_ground_truth(os.path.join(EXP_ROOT, 'cluster_label.csv'), TRAIN_KPI)
        y_pred, label_mappings = match_class_labels(y_true, self.y_pred)
        if self.ignore_outlier:
            evaluation_pr_ignore_outlier(y_true, y_pred)
        else:
            evaluation_pr(y_true, y_pred)

        # classify result evaluation
        test_true = get_ground_truth(os.path.join(EXP_ROOT, 'cluster_label.csv'), TEST_KPI)
        # test_pred = match_class_labels(test_true, self.test_pred)
        test_pred = match_test_labels(self.test_pred, label_mappings)
        if self.ignore_outlier:
            evaluation_pr_ignore_outlier(test_true, test_pred)
        else:
            evaluation_pr(test_true, test_pred)

        # total evaluation
        true = np.append(y_true, test_true)
        pred = np.append(y_pred, test_pred)
        if self.ignore_outlier:
            evaluation_pr_ignore_outlier(true, pred)
        else:
            evaluation_pr(true, pred)


