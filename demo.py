#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import multiprocessing
from examples.plotutils import delay_eva
import os
import config
import random
from logging import getLogger,basicConfig
from collections import OrderedDict, Counter
import random
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from frameworks.CPLELearning import CPLELearningModel
from frameworks.PULearning import PULearningModel
from sklearn.externals import joblib


result_root=config.DATA_ROOT+"/anomalyresult"
if(not(os.path.exists(result_root))):
    os.mkdir(result_root)

al_name = "PUAD"
model_root = config.DATA_ROOT+"/anomalyresult/model/"
if(not(os.path.exists(model_root))):
    os.mkdir(model_root)
model_root=config.DATA_ROOT+"/anomalyresult/model/"+al_name
if(not(os.path.exists(model_root))):
    os.mkdir(model_root)



proba_root = config.DATA_ROOT+"/anomalyresult/proba/"
if(not(os.path.exists(proba_root))):
    os.mkdir(proba_root)
proba_root = config.DATA_ROOT+"/anomalyresult/proba/"+al_name
if(not(os.path.exists(proba_root))):
    os.mkdir(proba_root)

random.seed()
source_train_ratio = 1
target_train_ratio = 0.4
target_test_ratio = 0.6
label_ratio = 0.1
max_label_round = 2


def cal_fscore(matrix):
    TP = float(matrix[1][1])
    fscore, pre, rec=0,0,0
    if TP!=0:
        FP = matrix[0][1]
        FN = matrix[1][0]
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        fscore=2*pre*rec/(pre+rec)
    return fscore,pre,rec


def save_proba(model, df, name, new_pre=None, ads_feature_name=None, PU_label=None):
    feature_name = [i for i in df.columns if i.startswith("F#")]
    if ads_feature_name != None:
        feature_name = ads_feature_name
    df_proba = model.predict_proba(df[feature_name])
    df_proba = df_proba[:, 1]
    df_save = df[["label"]].copy()
    df_save["proba"] = df_proba
    if new_pre is not None:
        df_save["predict"] = new_pre
    if PU_label is not None:
        df_save["PU_label"] = PU_label
    df_save.to_csv(os.path.join(proba_root, name), index=False)


def label_positive(train, label_count, type):
    positive_df = train[train['label'] == 1]
    positive_df_index = positive_df.index.tolist()
    if type  == 'random':
        random.shuffle(positive_df_index)
    if len(positive_df_index) <= label_count:
        label_count = len(positive_df_index)
    selected_positive_index = positive_df_index[:label_count]
    train.loc[:, 'label'] = -1
    train.loc[selected_positive_index, 'label'] = 1
    return train, label_count


def main(centroid_name, cluster_name, feature_root, centroid_KPI_label_count):
    matrix = np.array([[0, 0], [0, 0]])
    summary_cluster = []
    summary_curve = []
    df_source = pd.read_csv(os.path.join(feature_root, centroid_name))
    feature_name = [i for i in df_source.columns if i.startswith("F#")]
    cluster_matrix = np.array([[0, 0], [0, 0]])
    real_label = df_source['label'].values.copy()
    df_source, centroid_KPI_real_label_count = label_positive(df_source, centroid_KPI_label_count, 'random')
    centroid_train = df_source.copy()
    '''Start PU learning'''
    PU_model = PULearningModel(centroid_train[feature_name].values, centroid_train['label'].values, len(centroid_train))
    print Counter(centroid_train['label'].values)
    PU_model.pre_training(0.2)
    print Counter(real_label)
    RF_model = RandomForestClassifier(n_estimators=100)
    PU_labels, positive_label_count = PU_model.add_reliable_samples_using_RandomForest(0.015, 200, 0.7, real_label, RF_model)
    train_data = centroid_train[feature_name].values
    centroid_train['label'] = PU_labels
    print 'Finish PU learning for centroid:', Counter(centroid_train['label'].values)
    '''Finish PU learning'''

    for name_suffix in os.listdir(feature_root):
        if name_suffix == centroid_name:
            continue
        print('*'*30)
        print(name_suffix)
        print('*'*30)
        df_target = pd.read_csv(os.path.join(feature_root, name_suffix))

        target_test_length = int(target_test_ratio * len(df_target))
        test = df_target[-target_test_length:].copy()
        target_train_length = int(target_train_ratio * len(df_target))
        target_train = df_target[:target_train_length].copy()
        target_train_with_label = target_train.copy()
        target_train['label'] = -1

        train = pd.concat([centroid_train, target_train]).copy()
        print Counter(train['label'].values)
        model = CPLELearningModel(basemodel=RandomForestClassifier(config.RF_n_trees, n_jobs=15), max_iter=50,
                                    predict_from_probabilities=True, real_label=None)
        train_data = train[feature_name].values
        train_label = train['label'].values
        print 'start training CPLE model:', Counter(train_label)
        model.fit(train_data, train_label)
        print("finish train")
        # exit()
        name = name_suffix + '_PU'
        joblib.dump(model, model_root + '/' + name + ".sav")
        model1 = joblib.load(model_root + '/' + name + '.sav')
        print("model is :", model1)

        proba=model.predict_proba(test[feature_name])
        proba=proba[:, 1]

        eva = delay_eva(test["label"].values, proba)
        print(proba)
        _, best_threshold = eva.best_fscore_threshold()
        threshold = best_threshold
        print "threshold is", threshold

        predict_ans = eva.predict_for_threshold(threshold)
        save_proba(model, test, name + "_test" + ".csv", predict_ans)

        fscore = eva.fscore_for_threshold(threshold)
        average_detection_delay = eva.average_detection_delay(threshold)*config.interval/60
        print("PUAD fscore of test is %f",fscore)
        print("PUAD average_detection_delay is %f", average_detection_delay)
        temp_matrix = eva.confusion_matrix_for_threshold(threshold)
        matrix = matrix+temp_matrix
        cluster_matrix = cluster_matrix+temp_matrix

        _, pre, rec = cal_fscore(temp_matrix)
        TP = temp_matrix[1][1]
        FP = temp_matrix[0][1]
        FN = temp_matrix[1][0]
        print 'TP:', TP
        print 'FP:', FP
        print 'FN:', FN
        temp = OrderedDict([("name", name),
                            ("medios",0),
                            ("label", test["label"].values.sum()),
                            ("PU_fscore", fscore),
                            ("delay", average_detection_delay),
                            ("pre", pre),
                            ("rec", rec),
                            ("TP", TP),
                            ("FP", FP),
                            ("FN", FN),
                            ("centroid_KPI_label_count", centroid_KPI_label_count),
                            ("threshold", threshold)])
        summary_curve.append(temp)

        # 在这里上面的循环结束

    df_curveresults = pd.DataFrame(summary_curve)
    df_curveresults.to_csv(os.path.join(result_root, 'PUAD_%s_%d_%f_result.csv' %(cluster_name, centroid_KPI_label_count, 0.015)),
                           index=False)



def refresh_feature_path(i):
    cluster_name = config.CLUSTER_PREFIX + str(i)
    feature_root = os.path.join(config.DATA_ROOT, cluster_name)
    return cluster_name, feature_root


if __name__ == '__main__':
    centroid_dict = {
                     '9': 'CVT0JD2_AvgCost_3_feature.csv',
                     '8': '1H0SF52_AvgCost_0_feature.csv',
                     '7': 'b467d3e438e823dc91b8a8f9341105bc608bfb5336890ce26f77fae3tps_2_feature.csv',
                     '6': '5C9B0K2_AvgCost_3_feature.csv',
                     '5': '3GNQH42_AvgCost_0_feature.csv',
                     '4': '5_11_feature.csv',
                     '3': '2GMVF62_AvgCost_3_feature.csv',
                     '2': '5_29_feature.csv',
                     '1': '0_56_feature.csv'
    }
    pool = multiprocessing.Pool(processes=36)
    for index in range(1, 10):
        cluster_name, feature_root = refresh_feature_path(index)
        # pool.apply_async(main, (centroid_dict[str(index)], cluster_name, feature_root, 8))
        main(centroid_dict[str(index)], cluster_name, feature_root, 8)
