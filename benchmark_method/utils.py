# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    precision_recall_curve, 
    auc, 
    roc_curve, 
    average_precision_score, 
    matthews_corrcoef,
    )
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

from CS_NCA import CS_NCA
from canonical_resampling import *
from self_paced_ensemble import SelfPacedEnsemble
from canonical_ensemble import *


def init_model(method, base_estimator, n_estimators):
    '''return a model specified by "method".'''
    if method == 'SPEnsemble':
        model = SelfPacedEnsemble(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBoost':
        model = SMOTEBoost(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBagging':
        model = SMOTEBagging(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'RUSBoost':
        model = RUSBoost(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'UnderBagging':
        model = UnderBagging(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'Cascade':
        model = BalanceCascade(base_estimator = base_estimator, n_estimators = n_estimators)
    else:
        # Not the ensemble method
        model = Resample_classifier(base_estimator = base_estimator, method = method)
    return model


def init_base_classifier(base_classifier_type='DT'):
    if base_classifier_type == 'DT':
        base_classifier = sklearn.tree.DecisionTreeClassifier()
    elif base_classifier_type == 'KNN':
        base_classifier = sklearn.neighbors.KNeighborsClassifier()
    elif base_classifier_type == 'SVM':
        base_classifier = sklearn.svm.SVC(probability=True)
    elif base_classifier_type == 'MLP':
        base_classifier = MLPClassifier()
    return base_classifier


class DR_model():
    def __init__(self,
                 save_features=20,
                 nca_mode='No',
                 dataset_name=None):
        self._n_features = save_features
        self.nca_mode = nca_mode
        self.dataset_name = dataset_name

    def fit(self, X, y):
        imbalance_ratio = len(y) / sum(y) - 1
        if self.nca_mode == 'Load':
            self.DR = CS_NCA(low_dims=self._n_features, cost_par=imbalance_ratio, max_steps=500,
                             optimizer='gd', init_style="uniform", learning_rate=0.01, verbose=True)
            A = np.load('../results/' + self.dataset_name +'_' + str(X.shape[1]) + '-' + str(self._n_features) + '.npy')
            self.DR.load_model(A)
            #print("Cost sensitive - Neighbor Component Analysis Load!")
        elif self.nca_mode == 'Train':
            self.DR = CS_NCA(low_dims=self._n_features, cost_par=imbalance_ratio, max_steps=500,
                             optimizer='gd', init_style="uniform", learning_rate=0.01, verbose=True)
            self.DR.fit(X, y)
            #print("Cost sensitive - Neighbor Component Analysis Done!")
        elif self.nca_mode == 'Save':
            self.DR = CS_NCA(low_dims=self._n_features, cost_par=imbalance_ratio, max_steps=500,
                             optimizer='gd', init_style="uniform", learning_rate=0.01, verbose=True)
            self.DR.fit(X, y)
            self.DR.save_model('../results/' + self.dataset_name +'_' + str(X.shape[1]) + '-' + str(self._n_features))
            #print("Cost sensitive - Neighbor Component Analysis Done and Save!")
        elif self.nca_mode == 'No':
            self.DR = LocallyLinearEmbedding(n_components=self._n_features)
            self.DR = self.DR.fit(X, y)
            #print("Cost sensitive - Neighbor Component Analysis dont Use!")
        else:
            raise Error('No nca mode support: {}'.format(self.nca_mode))
        return self.DR

    def transform(self, X):
        X_new = self.DR.transform(X)
        return X_new


def load_dataset(dataset_name):
    """Util function that load training/test data from /data folder.

    Parameters
    ----------
    dataset_name : string
        Name of the target dataset.
        Train/test data are expected to save in .csv files with
        suffix _{train/test}.csv. Labels should be at the last column
        named with 'label'.

    Returns
    ----------
    X_train, y_train, X_test, y_test
        Pandas DataFrames / Series
    """
    df_train = pd.read_csv(f'../data/{dataset_name}_train.csv')
    X_train = df_train[df_train.columns.tolist()[:-1]]
    y_train = df_train['label']
    df_test = pd.read_csv(f'../data/{dataset_name}_test.csv')
    X_test = df_test[df_test.columns.tolist()[:-1]]
    y_test = df_test['label']
    return X_train.values, y_train.values, \
           X_test.values, y_test.values


def make_binary_classification_target(y, pos_label, verbose=False):
    '''Turn multi-class targets into binary classification targets.'''
    pos_idx = (y==pos_label)
    y[pos_idx] = 1
    y[~pos_idx] = 0
    if verbose:
        print ('Positive target:\t{}'.format(pos_label))
        print ('Imbalance ratio:\t{:.3f}'.format((y==0).sum()/(y==1).sum()))
    return y

def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    X_maj = X[y==0]; y_maj = y[y==0]
    X_min = X[y==1]; y_min = y[y==1]
    X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
        X_maj, y_maj, test_size=test_size, random_state=random_state)
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
        X_min, y_min, test_size=test_size, random_state=random_state)
    X_train = np.concatenate([X_train_maj, X_train_min])
    X_test = np.concatenate([X_test_maj, X_test_min])
    y_train = np.concatenate([y_train_maj, y_train_min])
    y_test = np.concatenate([y_test_maj, y_test_min])
    return  X_train, X_test, y_train, y_test

def imbalance_random_subset(X, y, size, random_state=None):
    '''Get random subset while guarantee same class distribution.'''
    _, X, _, y = imbalance_train_test_split(X, y, 
        test_size=size, random_state=random_state)
    return X, y

def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return average_precision_score(label, y_pred)

def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    f1s = 2 * (prec * reca) / (prec + reca)
    return max(f1s)

def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec*reca), 0.5)
    return max(gms)

def mcc_optim(label, y_pred):
    '''Compute optimal MCC score.'''
    mccs = []
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0+t*0.01] = 0
        y_pred_b[y_pred_b >= 0+t*0.01] = 1
        mcc = matthews_corrcoef(label, y_pred_b)
        mccs.append(mcc)
    return max(mccs)

def precision_at_recall(label, y_pred, recall):
    '''Compute precision at recall.'''
    prec, reca, _ = precision_recall_curve(label, y_pred)
    idx = np.searchsorted(-reca, -recall, 'right')
    return prec[idx - 1]

def recall_at_precision(label, y_pred, precision):
    '''Compute recall at precision.'''
    prec, reca, _ = precision_recall_curve(label, y_pred)
    idx = np.searchsorted(prec, precision, 'right')
    return reca[idx]

class Error(Exception):
    '''Simple exception.'''
    pass