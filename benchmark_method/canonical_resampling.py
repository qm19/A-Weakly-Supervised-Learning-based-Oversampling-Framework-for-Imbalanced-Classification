# -*- coding: utf-8 -*-

from imblearn.under_sampling import (
    ClusterCentroids, 
    NearMiss, 
    RandomUnderSampler, 
    EditedNearestNeighbours, 
    AllKNN, 
    TomekLinks, 
    OneSidedSelection, 
    RepeatedEditedNearestNeighbours, 
    CondensedNearestNeighbour, 
    NeighbourhoodCleaningRule,
)
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE,
)
from imblearn.combine import (
    SMOTEENN, SMOTETomek,
)

from sklearn.tree import DecisionTreeClassifier as DT
import pandas as pd

class Error(Exception):
    pass

class Resample_classifier(object):
    '''
    Re-sampling methods for imbalance classification, based on imblearn python package.
    imblearn url: https://github.com/scikit-learn-contrib/imbalanced-learn
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator used for training after re-sampling
    '''
    def __init__(self, base_estimator=DT(), method='RUS'):
        self.base_estimator = base_estimator
        self.method = method
    
    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    
    def fit(self, X, y, random_state=None, visualize=False):
        '''
        by: String
            The method used to perform re-sampling
            currently support: ['RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
                'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 'SMOTETomek',
                'ORG']
        '''
        by = self.method
        if by == 'RUS':
            sampler = RandomUnderSampler(random_state=random_state)
        elif by == 'CNN':
            sampler = CondensedNearestNeighbour(random_state=random_state)
        elif by == 'ENN':
            sampler = EditedNearestNeighbours()
        elif by == 'NCR':
            sampler = NeighbourhoodCleaningRule()
        elif by == 'Tomek':
            sampler = TomekLinks()
        elif by == 'ALLKNN':
            sampler = AllKNN()
        elif by == 'OSS':
            sampler = OneSidedSelection(random_state=random_state)
        elif by == 'NM':
            sampler = NearMiss()
        elif by == 'CC':
            sampler = ClusterCentroids(random_state=random_state)
        elif by == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif by == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
        elif by == 'BorderSMOTE':
            sampler = BorderlineSMOTE(random_state=random_state)
        elif by == 'SMOTEENN':
            sampler = SMOTEENN(random_state=random_state)
        elif by == 'SMOTETomek':
            sampler = SMOTETomek(random_state=random_state)
        elif by == 'ORG':
            sampler = None
        else:
            raise Error('Unexpected \'by\' type {}'.format(by))
        
        if by != 'ORG':
            X_train, y_train = sampler.fit_resample(X, y)
        else:
            X_train, y_train = X, y
        if visualize:
            df = pd.DataFrame(X_train)
            df['label'] = y_train
            df.plot.scatter(x=0, y=1, c='label', s=3, colormap='coolwarm', title='{} training set'.format(by))
        self.base_estimator.fit(X_train, y_train)