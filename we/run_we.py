import logging
import argparse

from naruto_skills.word_embedding import WordEmbedding
from naruto_skills.text_transformer import TextTransformer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('preprocessed_csv_file', type=str, help='Path to dataset csv file')
    parser.add_argument('--vocab_file', type=str, help='Path to file for saving vocabulary')
    parser.add_argument('--embedding_weights_file', type=str, help='Path to file for saving embedding weights')

    args = parser.parse_args()
    logging.info('All params')
    logging.info(args)
    logging.info('\n' * 3)

    w_e = WordEmbedding(args.preprocessed_csv_file, min_freq=2, embedding_size=300, interesting_column='mention')
    w_e.add_vocab([TextTransformer.PADDING, TextTransformer.OOV])
    w_e.save_it(args.embedding_weights_file, args.vocab_file)
