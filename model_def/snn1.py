import os
import logging
import pickle

import numpy as np
import tensorflow as tf


from model_def import ingradient


class SNN1:
    
    def __init__(self, text_transformer, we_weights_path):
        """

        :param text_transformer: TODO remove it, model def should not involve to text_transformer in any way
        """

        self.text_transformer = text_transformer

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_X, self.tf_y = ingradient.build_input_3()
            self.tf_embedding = ingradient.build_word_embeddings(pre_trained_matrix=np.load(we_weights_path))
            tf_logits = ingradient.inference_snn_1(tf_X=self.tf_X, tf_embedding=self.tf_embedding)

            self.tf_prob = tf_logits

            self.tf_predict = ingradient.build_predict_2(self.tf_prob, threshold=0.5)
            print('tf_predict shape', self.tf_predict.shape)

            self.tf_loss = ingradient.build_loss_v1(tf_logits=tf_logits, tf_y=self.tf_y)

            self.tf_optimizer, self.tf_global_step = ingradient.build_optimize_v1(self.tf_loss, learning_rate=0.0001)

        self.params_dict = dict(tf_X=self.tf_X.name,
                                tf_y=self.tf_y.name,
                                tf_embedding=self.tf_embedding.name,
                                tf_predict=self.tf_predict.name,
                                tf_optimizer=self.tf_optimizer.name,
                                tf_global_step=self.tf_global_step.name,
                                tf_predict_prob=self.tf_prob.name,
                                length=self.tf_X.shape[1].value/2
                                )
        self.params_dict.update({'feed_dict_for_infer_func': SNN1._feed_dict_for_infer,
                                 'feed_dict_for_train_func': SNN1._feed_dict_for_train,
                                 })

    @staticmethod
    def _feed_dict_for_infer(tf_X, X):
        x1, x2 = list(zip(*X))
        x1 = np.array([item['mention'] for item in x1])
        x2 = np.array([item['mention'] for item in x2])
        x = np.concatenate((x1, x2), axis=1)
        train_feed_dict = {tf_X: x}
        return train_feed_dict

    @staticmethod
    def _feed_dict_for_train(tf_X, tf_y, X, y):
        x1, x2 = list(zip(*X))
        x1 = np.array([item['mention'] for item in x1])
        x2 = np.array([item['mention'] for item in x2])
        x = np.concatenate((x1, x2), axis=1)
        feed_dict = {tf_X: x, tf_y: y}
        return feed_dict

    def create_train_feed_dict(self, X, y):
        return SNN1._feed_dict_for_train(self.tf_X, self.tf_y, X, y)

    def create_eval_feed_dict(self, X, y):
        return self.create_train_feed_dict(X, y)

    def save_vocab(self, dir_path):
        self.text_transformer.save_vocab(os.path.join(dir_path, 'vocab.txt'))

    def save_tf_name(self, path_to_file):
        with open(path_to_file, 'wb') as output_file:
            pickle.dump(obj=self.params_dict, file=output_file, protocol=3)
