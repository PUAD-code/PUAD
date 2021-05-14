# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
import config
#from statsmodels import api as sm  
import statsmodels.api as sm
__all__=["evaluator","delay_eva"]

class evaluator(object):
    def __init__(self,y_true,y_proba):
        self.y_true=y_true
        self.y_proba=y_proba

    def average_detection_delay(self, threshold):

        truth=np.copy(self.y_true)
        proba=np.copy(self.y_proba)
        proba = (proba >= threshold).astype(np.float32)

        index = np.where(truth == 1)[0]
        segment_end_index = (index[:-1] != index[1:] - 1)  # type: np.ndarray
        segment_end_index = np.append(segment_end_index, True)
        segment_start_index = np.mod(
            np.where(segment_end_index)[0] + 1, len(segment_end_index)
        )
        segment_end_index = index[segment_end_index]
        segment_start_index = index[segment_start_index]
        segment_end_index.sort()
        segment_start_index.sort()

        segment_count = len(segment_end_index)
        cumsum_delay = 0
        anomaly_count = 0
        for i in range(segment_count):
            start = segment_start_index[i]
            end = segment_end_index[i]
            sub_index = np.arange(start, end + 1)
            delay_points = np.where(proba[sub_index] == 1)[0]
            if len(delay_points) == 0:
                delay = 0
            else:
                delay = delay_points[0]
                anomaly_count += 1
            cumsum_delay += delay
        return cumsum_delay * 1.0 / max(1, anomaly_count)

    def internal_process_truth_proba_missing(self):
        # for each ground truth anomaly segment, we use the maximum prob
        # in `proba` as the probability of all points in that segment.
        truth = np.copy(self.y_true)
        proba = np.copy(self.y_proba)
        splits = np.where(truth[1:] != truth[:-1])[0] + 1
        is_anomaly = truth[0] == 1
        pos = 0
        for sp in splits:
            if is_anomaly:
                proba[pos: sp] = np.max(proba[pos: sp])
            is_anomaly = not is_anomaly
            pos = sp
        sp = len(truth)
        if is_anomaly:
            proba[pos: sp] = np.max(proba[pos: sp])

        return truth, proba

    def best_fscore_threshold(self):
        truth,proba=self.internal_process_truth_proba_missing()
        precision, recall, threshold = metrics.precision_recall_curve(truth, proba)
        fscore = self.compute_fscore(precision, recall)
        idx = np.argmax(fscore[:-1])
        return fscore[idx], threshold[idx]

    def fscore_for_threshold(self, threshold,delay=0):
        truth, proba = self.internal_process_truth_proba_missing()
        predict = (proba >= threshold).astype(np.int32)
        return metrics.f1_score(truth, predict)

    @staticmethod
    def compute_fscore(precision, recall):
        precision = np.asarray(precision, dtype=np.float64)
        recall = np.asarray(recall, dtype=np.float64)
        return np.where(
            (precision == 0) | (recall == 0),
            0.0,
            2. * np.exp(
                np.log(np.maximum(precision, 1e-8)) +
                np.log(np.maximum(recall, 1e-8)) -
                np.log(np.maximum(precision + recall, 1e-8))
            )
        )

class delay_eva(evaluator):
    def __init__(self,y_true,y_proba):
        super(delay_eva, self).__init__(
            y_true = y_true,
            y_proba = y_proba
        )

    @staticmethod
    def get_range_proba(truth,predict, delay=7):
        splits = np.where(truth[1:] != truth[:-1])[0] + 1
        is_anomaly = truth[0] == 1
        new_predict = np.array(predict)
        pos = 0

        for sp in splits:
            if is_anomaly:
                if 1 in predict[pos:min(pos + delay + 1, sp)]:
                    new_predict[pos: sp] = 1
                else:
                    new_predict[pos: sp] = 0
            is_anomaly = not is_anomaly
            pos = sp
        sp = len(truth)

        if is_anomaly:  # anomaly in the end
            if 1 in predict[pos: min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        return new_predict

    def confusion_matrix_for_threshold(self,threshold,delay=config.delay):
        predict = (self.y_proba >= threshold).astype(np.int32)
        truth = np.copy(self.y_true)
        new_predict = delay_eva.get_range_proba(truth, predict, delay)
        con=metrics.confusion_matrix(truth,new_predict)
        return con

    def fscore_for_threshold(self, threshold,delay=config.delay):
        predict= (self.y_proba >= threshold).astype(np.int32)
        truth = np.copy(self.y_true)
        new_predict = delay_eva.get_range_proba(truth,predict,delay)
        fscore = metrics.f1_score(truth, new_predict)
        return fscore

    def predict_for_threshold(self,threshold,delay=config.delay):
        predict = (self.y_proba >= threshold).astype(np.int32)
        truth = np.copy(self.y_true)
        new_predict = delay_eva.get_range_proba(truth, predict, delay)
        return new_predict


    def best_fscore_threshold(self):
        max_fscore=-1
        best_thre=-1

        min_value=np.min(self.y_proba)
        max_value=np.max(self.y_proba)

        thre=min_value+0.005
        while thre<max_value:
            fscore=self.fscore_for_threshold(thre,config.delay)
            if fscore>max_fscore:
                max_fscore=fscore
                best_thre=thre
            thre=thre+0.01
        return max_fscore,best_thre

    def elbow_point(self):
        sample=np.copy(self.y_proba)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, 1, num=200)
        y = ecdf(x)

        nPoints = len(x)
        allCoord = np.vstack((x, y)).T
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = np.argmax(distToLine)

        threhold=x[idxOfBestPoint]
        return threhold
