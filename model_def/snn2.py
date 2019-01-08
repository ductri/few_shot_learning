import os
import logging
import pickle

import numpy as np
import tensorflow as tf


from model_def import ingradient


class SNN2:
    
    def __init__(self, height, width):
        """
        For image, e.i omniplot dataset
        :param text_transformer: TODO remove it, model def should not involve to text_transformer in any way
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_X, self.tf_y = ingradient.build_input_4(height, width)

            tf_logits = ingradient.inference_snn_4(tf_X=self.tf_X)

            self.tf_prob = ingradient.build_predict_prob(tf_logits)

            self.tf_predict = ingradient.build_predict(self.tf_prob, threshold=0.5)
            logging.info('tf_predict shape: %s', self.tf_predict.shape)

            self.tf_loss = ingradient.build_loss_v1(tf_logits=tf_logits, tf_y=self.tf_y)

            self.tf_optimizer, self.tf_global_step = ingradient.build_optimize_v1(self.tf_loss, learning_rate=0.0001)

        self.params_dict = dict(tf_X=self.tf_X.name,
                                tf_y=self.tf_y.name,
                                tf_predict=self.tf_predict.name,
                                tf_optimizer=self.tf_optimizer.name,
                                tf_global_step=self.tf_global_step.name,
                                tf_predict_prob=self.tf_prob.name,
                                length=self.tf_X.shape[1].value/2
                                )
        self.params_dict.update({'feed_dict_for_infer_func': SNN2._feed_dict_for_infer,
                                 'feed_dict_for_train_func': SNN2._feed_dict_for_train,
                                 })

    @staticmethod
    def _feed_dict_for_infer(tf_X, X):
        x1, x2 = list(zip(*X))
        x = np.stack((x1, x2), axis=1)
        train_feed_dict = {tf_X: x}
        return train_feed_dict

    @staticmethod
    def _feed_dict_for_train(tf_X, tf_y, X, y):
        x1, x2 = list(zip(*X))
        x = np.stack((x1, x2), axis=1)
        feed_dict = {tf_X: x, tf_y: y.astype(np.float32)}
        return feed_dict

    def create_train_feed_dict(self, X, y):
        return SNN2._feed_dict_for_train(self.tf_X, self.tf_y, X, y)

    def create_eval_feed_dict(self, X, y):
        return self.create_train_feed_dict(X, y)

    def save_vocab(self, dir_path):
        pass

    def save_tf_name(self, path_to_file):
        if not os.path.isdir(os.path.dirname(path_to_file)):
            os.mkdir(os.path.dirname(path_to_file))
        with open(path_to_file, 'wb') as output_file:
            pickle.dump(obj=self.params_dict, file=output_file, protocol=3)

    @staticmethod
    def scoring_func(y_true, y_pred):
        pass