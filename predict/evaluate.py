import logging
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics

from model.data_for_train.dataset import DataWithLabel
from server.predictor import Predictor


def print_report(y_true, y_pred, lb_transformer, df_lb_info):
    logging.info('Number of ground true: %s', len(y_true))
    logging.info('Class distribution: %s', list(np.sum(y_true, axis=0)))

    logging.info('-'*50)
    logging.info('MEASURING ON LV 3')
    logging.info('Unweighted macro:')
    logging.info('Precision: %s', metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro'))
    logging.info('Recall: %s', metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro'))
    logging.info('F1: %s', metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro'))
    logging.info('')
    logging.info('Weighted macro:')
    logging.info('Precision: %s', metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    logging.info('Recall: %s', metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    logging.info('F1: %s', metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    logging.info('')
    logging.info('Micro:')
    logging.info('Precision: %s', metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro'))
    logging.info('Recall: %s', metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro'))
    logging.info('F1: %s', metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro'))
    logging.info('')

    logging.info('-' * 50)
    logging.info('MEASURING ON LV 2')

    logging.info('-' * 50)
    logging.info('MEASURING ON LV 1')


def print_report_2(y_true, y_pred, lb_transformer, df_lb_info):
    logging.info('y_true: %s', y_true[:3])
    logging.info('y_pred: %s', y_pred[:3])
    logging.info(metrics.classification_report(y_true=np.squeeze(y_true), y_pred=np.squeeze(y_pred), digits=4))


def get_predict(predictor, test_dataset, batch_size=256):
    logging.info('Test dataset info')
    dataset.show_info(is_logging=True)

    logging.info('Number of classes lv 1: %s', len(set(df_lb_info['att_name_lv0'])))
    logging.info('Number of classes lv 2: %s', len(set(df_lb_info['att_name'])))
    logging.info('Number of classes lv 3 (lv 2 with sentiment): %s', df_lb_info.shape[0])

    y_true = []
    y_pred = []
    my_iter = test_dataset.get_data_iterator(batch_size=batch_size, num_epochs=1, is_strictly_equal=False)
    for X, y in my_iter:
        y_pred.extend(predictor._predict(X))
        y_true.extend(y)
        logging.info('%s/%s', len(y_pred), test_dataset.y.shape[0])
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return y_pred, y_true


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_params_dict', type=str, help='Path to saved model (prefix)')
    parser.add_argument('--path_to_model', type=str, help='Path to saved model (prefix)')
    parser.add_argument('--path_to_lb_transformer', type=str, help='Path label transformer pickle file')
    parser.add_argument('--path_to_lb_info', type=str, help='Path to label info csv file')
    parser.add_argument('--path_to_data_npz_file', type=str, help='Path to npz file')
    args = parser.parse_args()
    logging.info('All params')
    logging.info(args)
    logging.info('\n' * 3)

    dataset = DataWithLabel.create_from_npz(args.path_to_data_npz_file, 'test')
    lb_transformer = pickle.load(open(args.path_to_lb_transformer, 'rb'))
    predictor = Predictor(path_to_params=args.path_to_params_dict, path_to_model=args.path_to_model, path_to_vocab=None)
    df_lb_info = pd.read_csv(args.path_to_lb_info)

    y_pred, y_true = get_predict(predictor, test_dataset=dataset)
    print_report_2(y_pred=y_pred, y_true=y_true, lb_transformer=lb_transformer, df_lb_info=df_lb_info)
