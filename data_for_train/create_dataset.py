import os
import logging
import ast
import argparse
import itertools

import pandas as pd

from naruto_skills.text_transformer import TextTransformer
from naruto_skills.dataset import Dataset


if __name__ == '__main__':
    # TODO it should output with vocab file
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('preprocessed_data_file', type=str, help='Path to dataset file')
    parser.add_argument('vocab_file', type=str, help='Path to file for saving vocabulary')
    parser.add_argument('--npz_dir', type=str, help='Path to directory for dumping dataset', default='')
    parser.add_argument('--max_length', type=int, help='Unit: word')
    parser.add_argument('--dataset_size', type=int, help='Dataset size')

    args = parser.parse_args()
    logging.info('All params: %s', args)

    df_all = pd.read_csv(args.preprocessed_data_file, lineterminator='\n')
    no_null_row = df_all['mention'].isnull().sum()
    if no_null_row > 0:
        logging.warning('There are %s null mentions. We are removing them', no_null_row)
        df_all.dropna(inplace=True, subset=['mention'])
    df_all.drop_duplicates(inplace=True, subset=['mention'])

    df_all['attribute_good_ids'] = df_all['attribute_good_ids'].map(lambda x: [str(item) for item in ast.literal_eval(x)] if isinstance(x, str) else [])

    text_transformer = TextTransformer.get_instance(args.vocab_file)
    df_all['mention'] = text_transformer.docs_to_index(list(df_all['mention']), max_length=args.max_length, min_length=args.max_length)
    df_all[df_all['mark_for_test']].to_csv(os.path.join(args.npz_dir, 'test.csv'), index=None)
    df_all = df_all[~df_all['mark_for_test']]

    all_attribute_ids = set(list(itertools.chain(*df_all['attribute_good_ids'])))
    dfs = [df_all[df_all['attribute_good_ids'].map(lambda x: att_id in x)] for att_id in all_attribute_ids]

    num_attributes = len(all_attribute_ids)
    pos_size = int(args.dataset_size / 3)
    pos_per_class = int(pos_size / num_attributes)
    neg_size = args.dataset_size - pos_size
    neg_per_class = int(neg_size / num_attributes)
    logging.info('There are %s classes', num_attributes)
    logging.info('There will be %s positive data points, %s mentions per class', pos_size, pos_per_class)
    logging.info('There will be %s negative data points', neg_size)

    pos_data = []
    for df_ in dfs[:-1]:
        pos_data.extend(list(zip(df_[['mention', 'id', 'attribute_good_ids']].sample(pos_per_class, replace=True).to_dict('records'),
                 df_[['mention', 'id', 'attribute_good_ids']].sample(pos_per_class, replace=True).to_dict('records'))))
    pos_data.extend(list(zip(dfs[-1][['mention', 'id', 'attribute_good_ids']].sample(pos_size - pos_per_class*(num_attributes-1), replace=True).to_dict('records'),
                      dfs[-1][['mention', 'id', 'attribute_good_ids']].sample(pos_size - pos_per_class*(num_attributes-1), replace=True).to_dict('records'))))
    assert len(pos_data) == pos_size

    neg_data = []
    all_attribute_ids = list(all_attribute_ids)
    for att_id in all_attribute_ids[:-1]:
        mentions1 = df_all[df_all['attribute_good_ids'].map(lambda x: att_id in x)][['mention', 'id', 'attribute_good_ids']].sample(neg_per_class, replace=True).to_dict('records')
        mentions2 = df_all[df_all['attribute_good_ids'].map(lambda x: att_id not in x)][['mention', 'id', 'attribute_good_ids']].sample(neg_per_class, replace=True).to_dict('records')
        neg_data.extend(list(zip(mentions1, mentions2)))
    mentions1 = df_all[df_all['attribute_good_ids'].map(lambda x: all_attribute_ids[-1] in x)][['mention', 'id', 'attribute_good_ids']].sample(
        neg_size - neg_per_class*(num_attributes-1), replace=True).to_dict('records')
    mentions2 = df_all[df_all['attribute_good_ids'].map(lambda x: all_attribute_ids[-1] not in x)][['mention', 'id', 'attribute_good_ids']].sample(
        neg_size - neg_per_class*(num_attributes-1), replace=True).to_dict('records')
    neg_data.extend(list(zip(mentions1, mentions2)))
    assert len(neg_data) == neg_size

    X = pos_data + neg_data
    y = [1] * len(pos_data) + [0]*len(neg_data)

    dataset = Dataset.create_from_entire_data(X=X, y=y, eval_prop=0.1, test_prop=0.1)
    dataset.show_info()
    path_to_save = args.npz_dir
    if path_to_save:
        dataset.dump(path_to_save)
