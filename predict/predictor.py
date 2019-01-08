import tensorflow as tf
import logging
import pickle
import numpy as np


class Predictor:
    def __init__(self, path_to_params, path_to_model):
        with open(path_to_params, 'rb') as input_file:
            self.params_dict = pickle.load(input_file)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph('{}.meta'.format(path_to_model))
                saver.restore(self.sess, path_to_model)
        logging.info('Restored saved model at %s', path_to_model)

    def predict(self, list_images_1, list_images_2):
        X = list(zip(list_images_1, list_images_2))
        return self._predict(X)

    def _predict(self, X):
        tf_predict = self.graph.get_tensor_by_name(self.params_dict['tf_predict'])
        feed_dict = self.__build_feed_dict(X)
        return np.squeeze(self.sess.run(tf_predict, feed_dict=feed_dict))

    def _predict_prob(self, X):
        tf_predict_prob = self.graph.get_tensor_by_name(self.params_dict['tf_predict_prob'])
        feed_dict = self.__build_feed_dict(X)
        return np.squeeze(self.sess.run(tf_predict_prob, feed_dict=feed_dict))

    def predict_prob(self, list_images_1, list_images_2):
        X = list(zip(list_images_1, list_images_2))
        return self._predict_prob(X)

    def __build_feed_dict(self, X):
        tf_X = self.graph.get_tensor_by_name(self.params_dict['tf_X'])
        feed_dict_func = self.params_dict['feed_dict_for_infer_func']
        feed_dict = feed_dict_func(tf_X, X)
        return feed_dict
