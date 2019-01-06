import os
import logging
import argparse

from sklearn.metrics import fbeta_score, f1_score

from naruto_skills.text_transformer import TextTransformer
from naruto_skills.trainer import Trainer

from model_def.snn1 import SNN1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='What kind of model to be trained. Current only options:'
                                                     'BASELINE, ATTENTION, GRU')
    parser.add_argument('dataset_file', type=str, help='Path to dataset file')
    parser.add_argument('--vocab', type=str, help='Path to file for saving vocabulary')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--num_epochs', type=int, help='Batch size', default=10)
    parser.add_argument('--eval_interval', type=int, help='Evaluation interval', default=100)
    parser.add_argument('--word_embedding_npy', type=str, help='Path to pre-trained word embedding', default='')
    parser.add_argument('--run_type', type=str, help='Path to pre-trained word embedding', default='NORMAL')
    parser.add_argument('--continuous_training_config', type=str, help='Path to config file', default='NORMAL')
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=-1)
    args = parser.parse_args()
    logging.info('All params')
    logging.info(args)
    logging.info('\n' * 3)

    scoring_func = f1_score

    if args.model_type == 'SNN1':
        text_transformer = TextTransformer.get_instance(args.vocab)

        def my_scoring_func(y_true, y_pred):
            beta = 0.5
            return fbeta_score(y_pred=y_pred, y_true=y_true, beta=beta)
        scoring_func = my_scoring_func
        model = SNN1(text_transformer=text_transformer, we_weights_path=args.word_embedding_npy)
        transform_pred_ = None
    else:
        raise Exception('Mode type is invalid')

    trainer = Trainer()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    trainer.run_train(model, args.dataset_file, args.batch_size, args.num_epochs, args.eval_interval, current_dir,
                      transform_pred_=transform_pred_, scoring_func=scoring_func)
