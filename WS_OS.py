import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import warnings

from CS_NCA import CS_NCA
from utils import sample_weight_compute
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

warnings.filterwarnings("ignore")


class WS_OS():
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=1,
                 n_features=10,
                 imbalanced_threshold=0.9,
                 pro_threshold=0.7,
                 ensemble_method='Boosting',
                 oversampling_method='WS-SMOTE',
                 nca_mode='Save',
                 dataset_name=None,
                 random_state=None):
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self._n_estimators = n_estimators
        self._n_features = n_features
        self.imbalanced_threshold = imbalanced_threshold
        self.pro_threshold = pro_threshold
        self.ensemble_method = ensemble_method
        self.oversampling_method = oversampling_method
        self.nca_mode = nca_mode
        self.dataset_name = dataset_name
        self._random_state = random_state

    def _fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator_).fit(X, y)

    def _fit_CS_NCA(self, X, y):
        imbalance_ratio = len(y) / sum(y) - 1
        nca = CS_NCA(low_dims=self._n_features, cost_par=imbalance_ratio, max_steps=200,
                     optimizer='gd', init_style="uniform", learning_rate=0.01, verbose=True)
        if self.nca_mode == 'No':
            A = np.ones([X.shape[1],X.shape[1]])
            nca.load_model(A)
            #print("Cost sensitive - Neighbor Component Analysis dont Use!")
        elif self.nca_mode == 'Load':
            A = np.load('./results/' + self.dataset_name +'_' + str(X.shape[1]) + '-' + str(self._n_features) + '.npy')
            nca.load_model(A)
            #print("Cost sensitive - Neighbor Component Analysis Load!")
        elif self.nca_mode == 'Train':
            nca.fit(X, y)
            #print("Cost sensitive - Neighbor Component Analysis Done!")
        elif self.nca_mode == 'Save':
            nca.fit(X, y)
            nca.save_model('./results/' + self.dataset_name +'_' + str(X.shape[1]) + '-' + str(self._n_features))
            #print("Cost sensitive - Neighbor Component Analysis Done and Save!")
        else:
            raise Error('No nca mode support: {}'.format(self.nca_mode))
        return nca

    def _bagging_ensemble(self, X_maj, y_maj, X_min, y_min):
        np.random.seed(self._random_state)
        IR_sampled = min(10, max(2, len(y_maj)/len(y_min)/self._n_estimators))
        n_save_maj = int(IR_sampled*len(y_min))
        idx = np.random.choice(len(X_maj), n_save_maj, replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])
        return X_train, y_train

    def _boosting_ensemble(self, X_maj, y_maj, X_min, y_min):
        n_save_maj = int(2 * len(X_maj) / self._n_estimators)
        idx = range(len(y_maj))
        idx_use = pd.DataFrame(idx).sample(n_save_maj,
                                           weights=sample_weight_compute(self._y_pred_maj, mu=0.95, sigma=0.2),
                                           random_state=self._random_state).values
        X_train = np.concatenate([X_maj[idx_use[:,0]], X_min])
        y_train = np.concatenate([y_maj[idx_use[:,0]], y_min])
        return X_train, y_train

    def _cascade_ensemble(self, X_maj, y_maj, X_min, y_min):
        ir = len(y_min) / len(y_maj)
        keep_fp_rate = np.power(ir, 1/(self._n_estimators-1))
        df_maj = pd.DataFrame(X_maj); df_maj['pred_proba'] = self._y_pred_maj
        df_maj = df_maj.sort_values(by='pred_proba', ascending=False)[:int(keep_fp_rate * len(df_maj) + 1)]
        X_maj = df_maj.drop(columns=['pred_proba']).values
        y_maj = np.zeros((X_maj.shape[0],))
        return X_maj, y_maj, X_min, y_min

    def _resampling(self, X_train, y_train):
        if self.oversampling_method == 'SMOTE':
            smo = SMOTE(sampling_strategy=1, random_state=self._random_state)
            X_train, y_train = smo.fit_resample(X_train, y_train)
        elif self.oversampling_method == 'BorderlineSMOTE':
            bsmo = BorderlineSMOTE(sampling_strategy=1, random_state=self._random_state, kind="borderline-1")
            X_train, y_train = bsmo.fit_resample(X_train, y_train)
        elif self.oversampling_method == 'ADASYN':
            ada = ADASYN(sampling_strategy=1, random_state=self._random_state)
            X_train, y_train = ada.fit_resample(X_train, y_train)
        else:
            raise Error('No oversampling method support: {}'.format(self.oversampling_method))
        return X_train, y_train

    def _WS_Oversampling(self, X_train, y_train):
        X_smote_new, y_smote_new = np.empty((0, X_train.shape[1])), np.empty((0,))
        while sum(y_smote_new) + sum(y_train) <= self.imbalanced_threshold * (len(y_train) - sum(y_train)):
            # Use oversampling method to generate positive samples
            X_smote, _ = self._resampling(X_train, y_train)
            X_smote = X_smote[X_train.shape[0]:X_smote.shape[0]]

            # Graph semi-supervised learning is used to determine the probability of generating a minority of samples
            low_LU = np.append(X_train, X_smote, axis=0)
            pij_mat = pairwise_distances(low_LU, squared=True)
            np.fill_diagonal(pij_mat, np.inf)
            pij_mat = np.exp(0.0 - pij_mat - logsumexp(0.0 - pij_mat, axis = 1)[:, None])
            # Set all distances less than the threshold to 0 to prevent floating point overflow
            pij_mat = np.where(pij_mat > 1.0e-5, pij_mat, 0)
            Y_unlabel = np.diag(pij_mat.sum(axis=0)[len(y_train):]) - pij_mat[len(y_train):,len(y_train):]
            Y_unlabel = np.linalg.pinv(Y_unlabel).dot(pij_mat[len(y_train):, :len(y_train)]).dot(y_train)
            '''
            The generated data belonging to a small number of samples whose 
            probability is less than the threshold value are excluded
            '''
            temp = X_smote[Y_unlabel >= self.pro_threshold, :]
            #temp_y = Y_unlabel[Y_unlabel >= self.pro_threshold]
            temp_y = np.ones((temp.shape[0],))
            X_smote_new = np.append(X_smote_new, temp, axis=0)
            y_smote_new = np.append(y_smote_new, temp_y, axis=0)

        y_smote_new = np.ones((len(y_smote_new),))
        # Merge the generated data with the original data
        X_train_new = np.concatenate((X_train, X_smote_new), axis=0)
        y_train_new = np.concatenate((y_train, y_smote_new), axis=0)
        return X_train_new, y_train_new

    def fit(self, X, y, label_maj=0, label_min=1):
        self.estimators_ = []

        # Get the cs-nca for dimension reduction.
        self.nca = self._fit_CS_NCA(X, y)
        X = self.nca.transform(X)

        # Initialize by spliting majority / minority set
        X_maj = X[y == label_maj];
        y_maj = y[y == label_maj]
        X_min = X[y == label_min];
        y_min = y[y == label_min]
        self._y_pred_maj = np.zeros((len(y_maj),))

        # Loop start
        if self.ensemble_method == 'Bagging':
            for i_estimator in range(self._n_estimators):
                # Bagging Ensemble
                X_train, y_train = self._bagging_ensemble(X_maj, y_maj, X_min, y_min)
                X_train, y_train = self._WS_Oversampling(X_train, y_train)
                self.estimators_.append(
                    self._fit_base_estimator(
                        X_train, y_train))
            return self

        elif self.ensemble_method == 'Boosting':
            for i_estimator in range(self._n_estimators):
                # Boosting Ensemble
                X_train, y_train = self._boosting_ensemble(X_maj, y_maj, X_min, y_min)
                X_train, y_train = self._WS_Oversampling(X_train, y_train)
                self.estimators_.append(
                    self._fit_base_estimator(
                        X_train, y_train))
                # update predicted probability
                n_clf = len(self.estimators_)
                y_pred_maj_last_clf = self.estimators_[-1].predict_proba(X_maj)[:, 1]
                self._y_pred_maj = (self._y_pred_maj * (n_clf - 1) + y_pred_maj_last_clf) / n_clf
            return self

        elif self.ensemble_method == 'Cascade':
            for i_estimator in range(self._n_estimators):
                # Cascade Ensemble
                X_maj, y_maj, X_min, y_min = self._cascade_ensemble(X_maj, y_maj, X_min, y_min)
                X_train, y_train = self._bagging_ensemble(X_maj, y_maj, X_min, y_min)
                X_train, y_train = self._WS_Oversampling(X_train, y_train)
                self.estimators_.append(
                    self._fit_base_estimator(
                        X_train, y_train))
                # update predicted result
                self._y_pred_maj = self.predict(X_maj)
            return self

        elif self.ensemble_method == 'No':
            X_train, y_train = self._WS_Oversampling(X, y)
            self.estimators_.append(
                self._fit_base_estimator(
                    X_train, y_train))
        else:
            raise Error('No such ensemble method support: {}'.format(self.ensemble_method))

    def predict_proba(self, X):
        if not X.shape[1] == self._n_features:
            X = self.nca.transform(X)
        y_pred = np.array(
            [model.predict(X) for model in self.estimators_]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred

    def predict(self, X):
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_pred_binarized

    def score(self, X, y):
        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])