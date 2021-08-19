from time import clock
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")

from utils import *
import argparse
from tqdm import trange


METHODS = ['RUS', 'ENN', 'Tomek', 'NM', 'SMOTE', 'BorderSMOTE', 'ADASYN',  'SMOTEENN', 'SMOTETomek',
           'RUSBoost', 'SMOTEBoost', 'UnderBagging', 'SMOTEBagging', 'Cascade', 'SPEnsemble']
'''
METHODS = ['SMOTEENN', 'SMOTETomek', 'UnderBagging', 'SMOTEBagging', 'Cascade', 'SPEnsemble']
'''
RANDOM_STATE = 42

def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Benchmark Method Experiment',
        usage='benchmark_main.py --oversampling_method <method> --base_classifier <method> --nca_mode <string> ' +
              '--dataset_name <string> --n_estimators <integer> --n_features <integer> --runs <integer> --save_excel <bool>'
    )

    parser.add_argument('--method', type=str, default='all',
                        choices=METHODS + ['all'], help='Name of ensmeble method')
    parser.add_argument('--base_classifier', type=str, default='DT', choices=['DT', 'KNN', 'SVM', 'MLP'],
                        help='Name of base classifier')
    parser.add_argument('--nca_mode', type=str, default='No', choices=['No', 'Load', 'Train', 'Save'],
                        help='The mode of CS-NCA, No: dont use; Load: load the training nca; Train: train a nca and dont save; Save: train a nca and save')
    parser.add_argument('--dataset_name', type=str, default='15_6', help='The filename of experiment dataset')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of base estimators')
    parser.add_argument('--n_features', type=int, default=20, help='Number of features after dimensionality reduction')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs')
    parser.add_argument('--save_excel', type=bool, default=False, help='Whether save the result to excel')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse()

    # Set base classifier
    base_classifier = init_base_classifier(base_classifier_type=args.base_classifier)

    # Load train/test data
    X_train, y_train, X_test, y_test = load_dataset(args.dataset_name)

    # Dimensionality Reduction Method
    dr = DR_model(save_features=args.n_features, nca_mode=args.nca_mode, dataset_name=args.dataset_name)
    dr.fit(X_train, y_train)
    X_train, X_test = dr.transform(X_train), dr.transform(X_test)

    # Train & Record
    method_list = METHODS if args.method=='all' else [args.method]
    mean_list, std_list = [], []
    for method in method_list:
        print('\nRunning method:\t{} of dataset {}.csv - {} estimators in {} independent run(s) ...'.format(
            method, args.dataset_name, args.n_estimators, args.runs))
        # print('Running ...')
        scores = []; times = []
        try:
            with trange(args.runs) as t:
                for _ in t:
                    model = init_model(
                        method=method,
                        n_estimators=args.n_estimators,
                        base_estimator=base_classifier,
                    )
                    start_time = clock()
                    model.fit(X_train, y_train)
                    times.append(clock()-start_time)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    scores.append([
                        auc_prc(y_test, y_pred),
                        f1_optim(y_test, y_pred),
                        gm_optim(y_test, y_pred),
                        mcc_optim(y_test, y_pred)
                    ])
        except KeyboardInterrupt:
            t.close()
            raise
        t.close
        
        # Print results to console
        print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
        print('------------------------------')
        print('Metrics:')
        df_scores = pd.DataFrame(scores, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'])
        for metric in df_scores.columns.tolist():
            print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))

        # Output the result as an excel.
        mean_list.append(
            [df_scores['AUCPRC'].mean(), df_scores['F1'].mean(), df_scores['G-mean'].mean(),
             df_scores['MCC'].mean()])
        std_list.append(
            [df_scores['AUCPRC'].std(), df_scores['F1'].std(), df_scores['G-mean'].std(), df_scores['MCC'].std()])

    df_mean = pd.DataFrame(mean_list, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'], index=METHODS)
    df_std = pd.DataFrame(std_list, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'], index=METHODS)
    if args.save_excel:
        writer = pd.ExcelWriter('../results/' + args.dataset_name + '_benchmark_MLP.xls')
        df_mean.to_excel(writer, sheet_name='mean', float_format='%.3f')
        df_std.to_excel(writer, sheet_name='std', float_format='%.3f')
        writer.save()
    return

if __name__ == '__main__':
    main()