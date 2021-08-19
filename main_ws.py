# -*- coding: utf-8 -*-
from time import clock
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings("ignore")

from WS_OS import WS_OS
from utils import *
import argparse

import sklearn.neural_network as nn

RANDOM_STATE = 42

def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Ensemble Method Experiment',
        usage='main_ws.py --oversampling_method <method> --ensemble_method <method> --nca_mode <string> ' +
              '--dataset_name <string> --n_estimators <integer> --n_features <integer> --runs <integer>'
        )

    parser.add_argument('--oversampling_method', type=str, default='SMOTE',
                        choices=['SMOTE', 'BorderlineSMOTE', 'ADASYN'],
                        help='Name of resampling method')
    parser.add_argument('--ensemble_method', type=str, default='Boosting', choices=['Boosting', 'Bagging', 'Cascade', 'No'],
                        help='Name of ensmeble method')
    parser.add_argument('--nca_mode', type=str, default='Load', choices=['No', 'Load', 'Train', 'Save'],
                        help='The mode of nca, No: don‘t use; Load: load the training nca; Train: train a nca and don’t save; Save: train a nca and save')
    parser.add_argument('--dataset_name', type=str, default='15_6', help='The filename of experiment dataset')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of base estimators')
    parser.add_argument('--n_features', type=int, default=20, help='Number of features saved by nca')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse()
    oversampling_method = args.oversampling_method
    dataset_name = args.dataset_name
    n_estimators = args.n_estimators
    runs = args.runs

    # Load train/test data
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)

    # Train & Record
    #mean_list, std_list = [], []
    print(
        '\nEnsemble method: {} \nResampling method: {} \nNCA mode: {} \nDataset: {}.csv - {} estimators in {} independent run(s) ...'
            .format(args.ensemble_method, oversampling_method, args.nca_mode, dataset_name, n_estimators, runs))
    scores = [];
    times = []
    for i in range(runs):
        model = WS_OS(base_estimator=sklearn.tree.DecisionTreeClassifier(),
                      n_estimators=n_estimators,
                      ensemble_method=args.ensemble_method,
                      oversampling_method=oversampling_method,
                      n_features=args.n_features,
                      nca_mode=args.nca_mode,
                      dataset_name=dataset_name,
                      imbalanced_threshold=0.9,
                      pro_threshold=0.7)
        start_time = clock()
        model.fit(X_train, y_train)
        times.append(clock() - start_time)
        y_pred = model.predict_proba(X_test)[:, 1]
        scores.append([
            auc_prc(y_test, y_pred),
            f1_optim(y_test, y_pred),
            gm_optim(y_test, y_pred),
            mcc_optim(y_test, y_pred)
        ])

    # Print results to console
    print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
    print('------------------------------')
    print('Metrics:')
    df_scores = pd.DataFrame(scores, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'])
    for metric in df_scores.columns.tolist():
        print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))



