import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from naruto_skills.dataset import Dataset
from naruto_skills.text_transformer import TextTransformer
from naruto_skills import graph_utils


class Trainer:
    def __init__(self):
        pass

    def run_train(self, model, dataset_path, batch_size, num_epochs, eval_interval, transform_pred_=None, scoring_func=None):
        """

        :param model:
        :param dataset_path:
        :param batch_size:
        :param num_epochs:
        :param eval_interval:
        :param transform_pred_: transform from index-based to text-based prediction
        :param scoring_func: The larger score is, the better model is. This function receive 2 params: y_true, y_pred
        :return:
        """
        current_dir = os.path.realpath(os.path.dirname(__file__))

        def save_meta_data(experiment_name_):
            path_for_vocab = os.path.join(current_dir, 'output', 'saved_models', model.__class__.__name__,
                                          experiment_name_)
            logging.info('Backup vocabs at %s', path_for_vocab)
            model.save_vocab(path_for_vocab)

            path_for_tf_name = os.path.join(current_dir, 'output', 'saved_models', model.__class__.__name__,
                                            experiment_name_, 'tensor_name.pkl')
            model.save_tf_name(path_for_tf_name)
            logging.info('Backup tensor name at %s', path_for_tf_name)

        if scoring_func is None:
            scoring_func = f1_score
        dataset = Dataset.create_from_npz(dataset_path)
        dataset.show_info()
        train_iterator = dataset.data_train.get_data_iterator(batch_size=batch_size, num_epochs=num_epochs, is_strictly_equal=False)

        with model.graph.as_default() as gr:
            tf_streaming_loss, tf_streaming_loss_op = tf.metrics.mean(values=model.tf_loss)
            tf_streaming_loss_summary = tf.summary.scalar('streaming_loss', tf_streaming_loss)

            experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
            save_meta_data(experiment_name)

            path_to_graph_train = os.path.join(current_dir, 'output', 'summary', 'train_' + experiment_name)
            path_to_graph_eval = os.path.join(current_dir, 'output', 'summary', 'eval_' + experiment_name)
            writer_train = tf.summary.FileWriter(logdir=path_to_graph_train, graph=model.graph)
            writer_eval = tf.summary.FileWriter(logdir=path_to_graph_eval, graph=model.graph)
            logging.info('Saved graph to %s', path_to_graph_train)

            writer_eval.flush()
            writer_train.flush()

            saver = tf.train.Saver(max_to_keep=5)

            logging.info('Model contains %s parameters', graph_utils.count_trainable_variables())
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            with tf.Session(graph=model.graph,
                            config=tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options)).as_default() as sess:
                # Export graph

                all_summary_op = tf.summary.merge_all()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                best_score = -1

                for X, y in train_iterator:
                    train_feed_dict = model.create_train_feed_dict(X, y)
                    _, global_step = sess.run([model.tf_optimizer, model.tf_global_step], feed_dict=train_feed_dict)
                    # logging.info('Step %s', global_step)
                    if global_step % 10 == 0:
                        train_loss = sess.run( model.tf_loss, feed_dict=train_feed_dict)
                        logging.info('Step: %s - Train loss: %s', global_step, train_loss)

                    if (global_step-1) % eval_interval == 0:
                        # Reset streaming metrics
                        sess.run(tf.local_variables_initializer())

                        summary_data = sess.run(all_summary_op, feed_dict=train_feed_dict)
                        writer_train.add_summary(summary_data, global_step=global_step)

                        # Reset streaming metrics
                        sess.run(tf.local_variables_initializer())
                        eval_iterator = dataset.data_eval.get_data_iterator(batch_size=batch_size, num_epochs=1)
                        y_pred = []
                        y_true = []
                        for X_eval, y_eval in eval_iterator:
                            y_true.extend(y_eval)
                            eval_feed_dict = model.create_train_feed_dict(X_eval, y_eval)
                            sess.run(tf_streaming_loss_op, feed_dict=eval_feed_dict)
                            pred = np.squeeze(sess.run(model.tf_predict, feed_dict=eval_feed_dict), axis=-1)
                            print('pred', pred)
                            y_pred.extend(pred)
                        y_pred = np.array(y_pred)
                        y_true = np.array(y_true)
                        eval_score = scoring_func(y_true=y_true, y_pred=y_pred)

                        summary_data = sess.run(tf_streaming_loss_summary)
                        writer_eval.add_summary(summary_data, global_step=global_step)

                        logging.info('Step: %s - Eval score: %s', global_step, eval_score)

                        if transform_pred_ is not None:
                            sample_dict_eval = model.create_train_feed_dict(X_eval[:3], y_eval[:3])
                            pred_eval = sess.run(model.tf_predict, feed_dict=sample_dict_eval)
                            sample_dict_train = model.create_train_feed_dict(X[:3], y[:3])
                            pred_train = sess.run(model.tf_predict, feed_dict=sample_dict_train)
                            logging.info(pred_train.shape)
                            logging.info('*' * 40)
                            logging.info('Predicting on train ...')
                            logging.info(transform_pred_(X[:3], y[:3], pred_train))
                            logging.info('*' * 40)
                            logging.info('Predicting on eval ...')
                            logging.info(transform_pred_(X_eval[:3], y_eval[:3], pred_eval))

                        if eval_score > best_score:
                            best_score = eval_score

                            path_to_model = os.path.join(current_dir, 'output', 'saved_models', model.__class__.__name__,
                                                         experiment_name)
                            saver.save(sess, save_path=path_to_model,
                                            global_step=global_step,
                                            write_meta_graph=True)
                            logging.info('Gained better score, saved model to %s', path_to_model)

                        logging.info('Best score: %s', best_score)
                        logging.info('\n')

                writer_train.close()
                writer_eval.close()


def get_transform_pred_func(vocab_file):
    text_transformer = TextTransformer.get_instance(vocab_file)

    def transform_pred(X, y, pred):
        """

        :param pred: batch_size, max_time
        :return:
        """
        text = '\n'
        for doc, predict, ground_truth in zip(text_transformer.index_to_docs(X, is_skip_padding=True),
                                              text_transformer.index_to_docs(pred, is_skip_padding=True),
                                              text_transformer.index_to_docs(y, is_skip_padding=True)):
                text += '* Src: %s\n' % doc
                text += '* Pre: %s\n' % predict
                text += '* Gro: %s\n' % ground_truth
                text += '\n'
        return text

    return transform_pred


def get_transform_pred_func_with_len(vocab_file):
    text_transformer = TextTransformer.get_instance(vocab_file)

    def transform_pred(X, y, pred):
        """

        :param pred: batch_size, max_time
        :return:
        """
        y, l = zip(*y)
        text = '\n'
        for doc, predict, ground_truth in zip(text_transformer.index_to_docs(X, is_skip_padding=True),
                                              text_transformer.index_to_docs(pred, is_skip_padding=True),
                                              text_transformer.index_to_docs(y, is_skip_padding=True)):
                text += '* Src: %s\n' % doc
                text += '* Pre: %s\n' % predict
                text += '* Gro: %s\n' % ground_truth
                text += '\n'
        return text

    return transform_pred
