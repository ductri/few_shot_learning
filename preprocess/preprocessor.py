import ast
import logging
import argparse
import pandas as pd

from nltk.tokenize import word_tokenize
import numpy as np

from preprocess import preprocess_supp


BRAND_REPLACE_CONST = 'tulanh_brand'
LIST_BRAND_KWS = {'Samsung', 'lg', 'toshiba'}


def __split_by_dot_tokenize(doc):
    return doc.replace('.', ' . ')


def __split_by_comma_tokenize(doc):
    return doc.replace(',', ' , ')


def __tokenize_single_doc(doc):
    return ' '.join(word_tokenize(doc))


def __cut_off(doc, length):
    return ' '.join(doc.split()[:length])


def __replace_brand_name(tokenized_doc):
    """

    :param tokenized_doc:
    :return:
    """
    tokenized_doc = ' '.join([BRAND_REPLACE_CONST if tok in LIST_BRAND_KWS else tok for tok in tokenized_doc.split()])
    return tokenized_doc


def preprocess_text(docs):
    docs = preprocess_supp.text_normalize(docs)
    docs = preprocess_supp.remove_html_tag(docs)
    docs = preprocess_supp.replace_url(docs, keep_url_host=False)
    docs = preprocess_supp.replace_phoneNB(docs)
    docs = preprocess_supp.remove_line_break(docs)
    docs = preprocess_supp.remove_special_chars(docs)
    docs = preprocess_supp.replace_all_number(docs)
    docs = preprocess_supp.replace_email(docs)
    docs = preprocess_supp.lowercase(docs)

    docs = [__split_by_dot_tokenize(doc) for doc in docs]
    docs = [__tokenize_single_doc(doc) for doc in docs]
    docs = [__replace_brand_name(doc) for doc in docs]

    logging.info('Pre-processing done')
    logging.info('-- Some samples: ')
    random_index = list(range(len(docs)))
    np.random.shuffle(random_index)
    for i in random_index[:15]:
        logging.info('-- -- %s', docs[i])

    return docs


def train_preprocess(docs, max_length):
    docs = preprocess_text(docs)
    docs = [__cut_off(doc, max_length) for doc in docs]
    return docs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', type=str,
                        help='Path to dataset file. This must be an CSV-format file without columns.'
                             'And the first column contains the text.')
    parser.add_argument('preprocessed_file', type=str, help='Path to file for preprocessed file')
    parser.add_argument('max_length', type=int)
    args = parser.parse_args()
    logging.info('All params: %s', args)

    df = pd.read_csv(args.dataset_file, lineterminator='\n')
    num_null_mention = df['mention'].isnull().sum()
    if num_null_mention > 0:
        logging.warning('There are %s null mentions', num_null_mention)
        df.dropna(subset=['mention'], inplace=True)
    df['attribute_ids'] = df['attribute_ids'].map(lambda x: [str(item) for item in ast.literal_eval(x)] if isinstance(x, str) else [])

    logging.info('Preprocessing file with %s mentions', df.shape[0])
    df['mention'] = train_preprocess(df['mention'], args.max_length)

    df.to_csv(args.preprocessed_file, index=None)
    logging.info('Saved preprocessed file to %s', args.preprocessed_file)
