import os
import logging
import ast
import argparse
import itertools

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from naruto_skills.dataset import Dataset


def load_image_with_label(path_to_omniglot_dir):
    images = []
    names = []
    list1 = os.listdir(path_to_omniglot_dir)
    logging.debug('There are %s alphabets', len(list1))
    for alphabet_dir_name in list1:
        path_alphabet = os.path.join(path_to_omniglot_dir, alphabet_dir_name)
        list2 = os.listdir(path_alphabet)
        logging.debug('There are %s characters within alphabet %s', len(list2), alphabet_dir_name)
        for character_dir_name in list2:
            path_character = os.path.join(path_alphabet, character_dir_name)
            list3 = os.listdir(path_character)
            logging.debug('There are %s instances within characters %s', len(list3), character_dir_name)
            for instance_name in list3:
                path_instance = os.path.join(path_character, instance_name)
                image = Image.open(path_instance)
                image = np.asarray(image).astype(np.float32)
                image = image / image.std() - image.mean()
                images.append(image)
                names.append('/'.join(path_instance.split('/')[-3:]))
    df = pd.DataFrame({'image': images, 'name': names})
    df['alphabet'] = df['name'].map(lambda x: x.split('/')[0])
    df['character'] = df['name'].map(lambda x: x.split('/')[1])
    df['instance'] = df['name'].map(lambda x: x.split('/')[2])
    df['class'] = df['name'].map(lambda x: '/'.join(x.split('/')[:2]))
    return df


def create_pair_dataset(df, size):
    pos_size = int(size/2)
    neg_size = size - pos_size

    list_classes = list(set(list(df['class'])))
    num_classes = len(list_classes)

    no_pos_per_class = [int(pos_size/num_classes)]*(num_classes-1) + [int(pos_size/num_classes) + (pos_size%num_classes)]
    no_neg_per_class = [int(neg_size / num_classes)] * (num_classes - 1) + [int(neg_size / num_classes) + (neg_size % num_classes)]
    pos_data = []
    for class_name, no_samples in zip(list_classes, no_pos_per_class):
        list1 = df[df['class'] == class_name]['image'].sample(no_samples, replace=True)
        list2 = df[df['class'] == class_name]['image'].sample(no_samples, replace=True)
        pos_data.extend(list(zip(list1, list2)))

    neg_data = []
    for class_name, no_samples in zip(list_classes, no_neg_per_class):
        list1 = df[df['class'] == class_name]['image'].sample(no_samples, replace=True)
        list2 = df[df['class'] != class_name]['image'].sample(no_samples, replace=True)
        neg_data.extend(list(zip(list1, list2)))
    X = pos_data + neg_data
    y = [1] * len(pos_data) + [0]*len(neg_data)
    return X, y


if __name__ == '__main__':
    # TODO it should output with vocab file
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to dataset file')
    parser.add_argument('--npz_dir', type=str, help='Path to directory for dumping dataset', default='')
    parser.add_argument('--dataset_size', type=int, help='Dataset size')
    parser.add_argument('--name', type=str, help='Dataset name')

    args = parser.parse_args()
    logging.info('All params: %s', args)

    df = load_image_with_label(args.data_dir)
    X, y = create_pair_dataset(df, size=args.dataset_size)

    my_dataset = Dataset.create_from_entire_data(X, y, eval_prop=0.1, test_prop=0.0)
    my_dataset.dump(args.npz_dir)
