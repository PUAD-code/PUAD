from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import RidgeClassifier, ElasticNet
from imblearn.over_sampling import SMOTE
from collections import Counter


class PULearningModel(BaseEstimator):
    def __init__(self, samples, labels, length):
        self.samples = samples
        self.length = length
        self.labels = labels

    def pre_training(self, pre_training_ratio):
        # clf = SVC(kernel='linear', probability=True, class_weight='balanced')
        # clf = SGDClassifier(loss='log', max_iter=10000)
        # clf = LogisticRegression(max_iter=10000)
        clf = ElasticNet(max_iter=10000)
        # clf = RandomForestClassifier()
        self.smote_for_positive()
        positive_index = np.where(self.labels == 1)
        self.labels = [0 if label == -1 else label for label in self.labels]
        # verify the count of labeling negative
        label_negative_count = int(pre_training_ratio * np.bincount(self.labels)[0])
        # fit pre-training model
        clf.fit(self.samples, self.labels)
        # get and sort distances
        distances = clf._decision_function(self.samples)
        # proba = clf.predict_proba(self.samples)
        # print [distances[index] for index in positive_index]
        distances_copy = distances.copy()
        distances_copy.sort()
        # get threshold
        negative_threshold = distances_copy[label_negative_count]
        self.labels = np.array([0 if distance < negative_threshold else -1 for distance in distances])
        self.labels[positive_index] = 1
        return self.labels[:self.length]
    
    def pre_training_without_SMOTE(self, pre_training_ratio):
        clf = ElasticNet(max_iter=10000)
        # self.smote_for_positive()
        positive_index = np.where(self.labels == 1)
        self.labels = [0 if label == -1 else label for label in self.labels]
        # verify the count of labeling negative
        label_negative_count = int(pre_training_ratio * np.bincount(self.labels)[0])
        # fit pre-training model
        clf.fit(self.samples, self.labels)
        # get and sort distances
        distances = clf._decision_function(self.samples)
        distances_copy = distances.copy()
        distances_copy.sort()
        # get threshold
        negative_threshold = distances_copy[label_negative_count]
        self.labels = np.array([-1 if label == 0 else label for label in self.labels])
        label_negative_count_cur = 0
        for i in range(self.length):
            if label_negative_count_cur >= label_negative_count:
                break
            if distances[i] <= negative_threshold and i not in positive_index[0]:
                self.labels[i] = 0
                label_negative_count_cur += 1
        print Counter(self.labels)
        return self.labels[:self.length]

    # label some samples
    def add_reliable_samples(self, class_prior, speed, add_ratio, real_label, model=RandomForestClassifier(n_estimators=100)):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) == 0:
                break
            '''
                random forest
            '''
            clf = model
            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf.predict_proba(unlabeled_set)
            prob_list = np.array([prob[1] for prob in prob_list])
            '''
                linear model
            '''
            clf_2 = ElasticNet(max_iter=10000)
            clf_2.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            dis_list = clf_2._decision_function(unlabeled_set)
            
            distance_list = prob_list.tolist()
            distance_list.sort()

            '''get the index of positive and negative index in unlabeled samples'''
            negative_threshold = distance_list[negative_count_each_round - 1]
            positive_threshold = distance_list[-positive_count_each_round]
            
            '''use linear model select samples'''
            select_dis_list = dis_list[np.where(prob_list <= negative_threshold)].tolist()
            select_dis_list.sort()
            
            negative_dis_threshold = select_dis_list[negative_count_each_round - 1]
            
            select_dis_list = dis_list[np.where(prob_list >= positive_threshold)].tolist()
            select_dis_list.sort()
            
            positive_dis_threshold = select_dis_list[-positive_count_each_round]
            
            positive_count_cur = 0
            negative_count_cur = 0
            
            for i in range(len(prob_list)):
                if prob_list[i] <= negative_threshold and dis_list[i] <= negative_dis_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    negative_count_cur += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and dis_list[i] >= positive_dis_threshold and positive_count_cur < positive_count_each_round:
                    label_count += 1
                    positive_label_count += 1
                    positive_count_cur += 1
                    self.labels[unlabeled_index[i]] = real_label[unlabeled_index[i]]
                    print real_label[unlabeled_index[i]]
                if label_count > max_label_count:
                    break
            print max(distance_list), min(distance_list)
            print label_count, positive_label_count
        print 'finish add reliable samples'
        return self.labels[:self.length], positive_label_count
    
    # label some samples
    def add_reliable_samples_using_LinearModel(self, class_prior, speed, add_ratio, real_label, model):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) == 0:
                break
            clf = model
            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf.decision_function(unlabeled_set)
            prob_list_copy = prob_list.copy()
            prob_list_copy.sort()
            
            negative_threshold = prob_list_copy[negative_count_each_round - 1]            
            positive_threshold = prob_list_copy[-positive_count_each_round]

            negative_count_cur = 0
            positive_count_cur = 0
            for i in range(len(prob_list)):
                if prob_list[i] <= negative_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    negative_count_cur += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and positive_count_cur < positive_count_each_round:
                    label_count += 1
                    positive_label_count += 1
                    positive_count_cur += 1
                    self.labels[unlabeled_index[i]] = real_label[unlabeled_index[i]]
                    print real_label[unlabeled_index[i]]
                if label_count > max_label_count:
                    break
            print max(prob_list), min(prob_list)
            print label_count, positive_label_count
        print 'finish add reliable samples'
        return self.labels[:self.length], positive_label_count

    # label some samples
    def add_reliable_samples_using_ElasticNet(self, class_prior, speed, add_ratio, real_label, model):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) == 0:
                break
            clf = model
            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf._decision_function(unlabeled_set)
            prob_list_copy = prob_list.copy()
            prob_list_copy.sort()
            
            negative_threshold = prob_list_copy[negative_count_each_round - 1]            
            positive_threshold = prob_list_copy[-positive_count_each_round]

            negative_count_cur = 0
            positive_count_cur = 0
            
            for i in range(len(prob_list)):
                if prob_list[i] <= negative_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    negative_count_cur += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and positive_count_cur < positive_count_each_round:
                    label_count += 1
                    positive_label_count += 1
                    positive_count_cur += 1
                    self.labels[unlabeled_index[i]] = real_label[unlabeled_index[i]]
                    print real_label[unlabeled_index[i]]
                if label_count > max_label_count:
                    break
            print max(prob_list), min(prob_list)
            print label_count, positive_label_count
        print 'finish add reliable samples'
        return self.labels[:self.length], positive_label_count
    
    def add_reliable_samples_using_RandomForest(self, class_prior, speed, add_ratio, real_label, model, if_smote=True):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            if if_smote == True:
                self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) == 0:
                break
            clf = RandomForestClassifier( n_estimators=100)
            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf.predict_proba(unlabeled_set)
            prob_list = np.array([prob[1] for prob in prob_list])
            prob_list_copy = prob_list.copy()
            prob_list_copy.sort()
            
            negative_threshold = prob_list_copy[negative_count_each_round - 1]            
            positive_threshold = prob_list_copy[-positive_count_each_round]
            
            negative_count_cur = 0
            positive_count_cur = 0
            for i in range(len(prob_list)):
                if prob_list[i] <= negative_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    negative_count_cur += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and positive_count_cur < positive_count_each_round and positive_label_count < int(class_prior * self.length):
                    label_count += 1
                    positive_label_count += 1
                    positive_count_cur += 1
                    self.labels[unlabeled_index[i]] = real_label[unlabeled_index[i]]
                    print real_label[unlabeled_index[i]]
                if label_count > max_label_count:
                    break
            print label_count, positive_label_count
        print 'finish add reliable samples'
        return self.labels[:self.length], positive_label_count

    def add_reliable_samples_without_label(self, class_prior, speed, add_ratio):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) == 0:
                break

            '''
                random forest
            '''
            clf = RandomForestClassifier( n_estimators=100)
            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf.predict_proba(unlabeled_set)
            prob_list = np.array([prob[1] for prob in prob_list])
            prob_list_copy = prob_list.copy()
            prob_list_copy.sort()
            negative_threshold = prob_list_copy[negative_count_each_round - 1]            
            positive_threshold = prob_list_copy[-positive_count_each_round]
            
            negative_count_cur = 0
            positive_count_cur = 0
            
            for i in range(len(prob_list)):
                if prob_list[i] <= negative_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    negative_count_cur += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and positive_count_cur < positive_count_each_round:
                    label_count += 1
                    positive_label_count += 1
                    positive_count_cur += 1
                    self.labels[unlabeled_index[i]] = 1
                if label_count > max_label_count:
                    break
            print label_count, positive_label_count
        print 'finish add reliable samples'
        return self.labels[:self.length], positive_label_count

    def add_reliable_samples_with_uncertainty(self, class_prior, speed, add_ratio, real_label):
        max_label_count = int(add_ratio * self.length)
        label_count = 0
        positive_label_count = 0
        positive_count_each_round = int(speed * class_prior)
        negative_count_each_round = int(speed * (1 - class_prior))
        while label_count < max_label_count:
            self.smote_for_positive()
            labeled_index = np.where(self.labels != -1)[0]
            unlabeled_index = np.where(self.labels == -1)[0]
            labeled_set = self.samples[labeled_index]
            unlabeled_set = self.samples[unlabeled_index]
            if len(unlabeled_set) <= speed:
                break

            '''
                random forest
            '''
            clf = RandomForestClassifier( n_estimators=100)

            clf.fit(labeled_set, self.labels[np.where(self.labels != -1)])
            prob_list = clf.predict_proba(unlabeled_set)
            prob_list = np.array([prob[1] for prob in prob_list])
            prob_list_copy = prob_list.copy()
            prob_list_copy.sort()
            negative_threshold = prob_list_copy[negative_count_each_round - 1]            
            positive_threshold = prob_list_copy[-positive_count_each_round]
            
            negative_count_cur = 0
            positive_count_cur = 0
            label_count_cur = 0
            
            '''active learning label most uncertain samples'''
            certainty_list = np.abs(prob_list - 0.5)
            sort_certainty_list = certainty_list.tolist()
            sort_certainty_list.sort()
            certainty_threshold = sort_certainty_list[int(class_prior * speed) - 1]
            for i in range(len(prob_list)):

                if certainty_list[i] <= certainty_threshold and label_count_cur < positive_count_each_round:
                    positive_label_count += 1
                    self.labels[unlabeled_index[i]] = real_label[unlabeled_index[i]]
                    label_count += 1
                    continue
                if prob_list[i] <= negative_threshold and negative_count_cur < negative_count_each_round:
                    label_count += 1
                    self.labels[unlabeled_index[i]] = 0
                elif prob_list[i] >= positive_threshold and positive_count_cur < positive_count_each_round:
                    label_count += 1
                    self.labels[unlabeled_index[i]] = 1
                if label_count > max_label_count:
                    break

        print 'finish add reliable samples with uncertainty'
        print positive_label_count
        return self.labels[:self.length], positive_label_count
    

    def smote_for_positive(self):
        k = 1
        m = 1
        if sum(self.labels[:self.length] == 1) > 10:
            k = 6
            m = 10
        smo = SMOTE(random_state=42, k_neighbors=k, m_neighbors=m)
        if 0 not in self.labels:
            x_smo, y_smo = smo.fit_sample(self.samples, self.labels)
            self.samples = np.concatenate((self.samples, x_smo[self.length:]), axis=0)
            self.labels = np.concatenate((self.labels, y_smo[self.length:]), axis=0)
        else:
            self.samples = self.samples[:self.length]
            self.labels = self.labels[:self.length]
            index = np.where(self.labels != -1)[0]
            x_smo, y_smo = smo.fit_sample(self.samples[index], self.labels[index])
            self.samples = np.concatenate((self.samples, x_smo[len(index):]), axis=0)
            self.labels = np.concatenate((self.labels, y_smo[len(index):]), axis=0)
