import logging

import tensorflow as tf
import numpy as np


def build_input():
    """
    Return tensor input
    """
    tf_X = tf.placeholder(dtype=tf.int32, name='tf_X', shape=[None, None])
    tf_y = tf.placeholder(dtype=tf.float32, name='tf_y', shape=[None])
    tf_sequence_len = tf.placeholder(tf.int32, name='tf_sequence_len', shape=[None])
    return tf_X, tf_y, tf_sequence_len


def build_input_2():
    """
    Return tensor input
    """
    tf_X = tf.placeholder(dtype=tf.int32, name='tf_X', shape=[None, 80])
    tf_y = tf.placeholder(dtype=tf.float32, name='tf_y', shape=[None])

    return tf_X, tf_y


def build_input_3():
    """
    Return tensor input
    """
    tf_X = tf.placeholder(dtype=tf.int32, name='tf_X', shape=[None, 200])
    tf_y = tf.placeholder(dtype=tf.float32, name='tf_y', shape=[None])

    return tf_X, tf_y


def build_input_4(height, width):
    """
    Return tensor input
    """
    tf_X = tf.placeholder(dtype=tf.float32, name='tf_X', shape=[None, 2, height, width])
    tf_y = tf.placeholder(dtype=tf.float32, name='tf_y', shape=[None])

    return tf_X, tf_y


def build_word_embeddings(pre_trained_matrix=None, vocab_size=None, embedding_size=None):
    with tf.device('/cpu:0'), tf.variable_scope('embedding_creation'):
        if pre_trained_matrix is None:
            return tf.get_variable(name='word_embeddings', dtype=tf.float32,
                                   shape=[vocab_size, embedding_size])
        else:
            return tf.get_variable(name='word_embeddings', dtype=tf.float32,
                                   initializer=pre_trained_matrix.astype(np.float32))


def inference(tf_X, tf_embedding, hidden_size, sequence_len):
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        # (batch_size, max_time, embedding_size)
        tf_projected_sens = tf.nn.embedding_lookup(params=tf_embedding, ids=tf_X)

    tf_encode_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name='tf_encode_fw_cell')

    tf_encode_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name='tf_encode_bw_cell')

    # TODO time major
    _, outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw=tf_encode_fw_cell,
                                                 cell_bw=tf_encode_bw_cell,
                                                 inputs=tf_projected_sens,
                                                 sequence_length=sequence_len,
                                                 dtype=tf.float32)
    # (batch_size, 2*hidden_size)
    tf_output = tf.concat((outputs[0].h, outputs[1].h), axis=-1)
    #
    tf_logits = tf.layers.dense(tf_output, 10, activation=tf.nn.relu)

    # batch_size, max_time, vocab_size
    tf_logits = tf.layers.dense(tf_logits, 1, name='tf_logits')
    print('tf_logits', tf_logits.shape)
    return tf_logits


def inference_2(tf_X, tf_embedding, hidden_size, sequence_len):
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        # (batch_size, max_time, embedding_size)
        tf_projected_sens = tf.nn.embedding_lookup(params=tf_embedding, ids=tf_X)
    tf_encode_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name='tf_encode_fw_cell')

    tf_encode_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name='tf_encode_bw_cell')

    # TODO time major
    _, outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw=tf_encode_fw_cell,
                                                 cell_bw=tf_encode_bw_cell,
                                                 inputs=tf_projected_sens,
                                                 sequence_length=sequence_len,
                                                 dtype=tf.float32)
    # (batch_size, 2*hidden_size)
    tf_output = tf.concat((outputs[0].h, outputs[1].h), axis=-1)
    #
    tf_logits = tf.layers.dense(tf_output, 250, activation=tf.nn.relu)

    # batch_size, max_time, vocab_size
    tf_logits = tf.layers.dense(tf_logits, 1, name='tf_logits')
    print('tf_logits', tf_logits.shape)
    return tf_logits


def inference_cnn_1(tf_X, tf_embedding):
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        # (batch_size, max_time, embedding_size)
        tf_projected_sens = tf.nn.embedding_lookup(params=tf_embedding, ids=tf_X)
        tf_projected_sens = tf.expand_dims(tf_projected_sens, axis=2)

    tf_inner = tf.layers.conv2d(inputs=tf_projected_sens, filters=100, kernel_size=(3, 1),
                                strides=(1, 1), activation=tf.nn.relu, padding='same')

    tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=80, kernel_size=(5, 1),
                                strides=(1, 1), activation=tf.nn.relu, padding='same')

    tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=50, kernel_size=(7, 1),
                                strides=(1, 1), activation=tf.nn.relu, padding='same')

    tf_inner = tf.layers.max_pooling2d(inputs=tf_inner, pool_size=(3, 1), strides=(1, 1), padding='same')

    tf_inner = tf.layers.flatten(tf_inner)

    # batch_size, max_time, vocab_size
    tf_logits = tf.layers.dense(tf_inner, 1, name='tf_logits')
    print('tf_logits', tf_logits.shape)
    return tf_logits


def inference_snn_1(tf_X, tf_embedding):
    tf_X1 = tf_X[:, :int(tf_X.shape[1].value / 2)]
    tf_X2 = tf_X[:, int(tf_X.shape[1].value / 2):]
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        # (batch_size, max_time, embedding_size)
        tf_projected_sens_1 = tf.nn.embedding_lookup(params=tf_embedding, ids=tf_X1)
        tf_projected_sens_1 = tf.expand_dims(tf_projected_sens_1, axis=2)
        tf_projected_sens_2 = tf.nn.embedding_lookup(params=tf_embedding, ids=tf_X2)
        tf_projected_sens_2 = tf.expand_dims(tf_projected_sens_2, axis=2)

    def sharing_network(tf_projected_sens):
        with tf.variable_scope('sharing_network'):
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

            tf_inner = tf.layers.conv2d(inputs=tf_projected_sens, filters=128, kernel_size=(3, 1),
                                        strides=(1, 1), activation=tf.nn.relu, padding='same', name='0',
                                        reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=regularizer)
            for i in range(3):
                tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=128, kernel_size=(5, 1),
                                            strides=(1, 1), activation=tf.nn.relu, padding='same',
                                            name='conv' + str(i + 1), reuse=tf.AUTO_REUSE,
                                            kernel_regularizer=regularizer)
                tf_inner = tf.layers.batch_normalization(tf_inner, name='batch_norm' + str(i + 1), reuse=tf.AUTO_REUSE)

            tf_inner = tf.layers.max_pooling2d(inputs=tf_inner, pool_size=(3, 1), strides=(1, 1), padding='same',
                                               name=str(i + 1))
            tf_inner = tf.layers.batch_normalization(tf_inner, name='batch_norm_last', reuse=tf.AUTO_REUSE)

            tf_inner = tf.layers.flatten(tf_inner, name='step_mostly_final')
            tf_inner = tf.layers.dense(tf_inner, units=4096, name='step_final', reuse=tf.AUTO_REUSE,
                                       kernel_regularizer=regularizer)
            return tf_inner

    tf_encoding_1 = sharing_network(tf_projected_sens_1)
    tf_encoding_2 = sharing_network(tf_projected_sens_2)
    tf_diff = tf.abs(tf.subtract(tf_encoding_1, tf_encoding_2))
    #
    # def get_distance(vec1, vec2):
    #     """
    #
    #     :param vec1: batch_size, hidden_size
    #     :param vec2: batch_size, hidden_size
    #     :return:
    #     """
    #     temp = tf.abs(tf.subtract(vec1, vec2))
    #     with tf.variable_scope('metric_weight'):
    #         tf_metric_w = tf.get_variable(name='metric_w', shape=(temp.shape[1]), dtype=tf.float32)
    #     return tf.reduce_sum(tf.multiply(temp, tf_metric_w), axis=-1)
    tf_logits = tf.layers.dense(tf_diff, units=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))

    logging.info('tf_logits: %s', tf_logits)
    return tf_logits


def inference_snn_2(tf_X):
    tf_X1 = tf.expand_dims(tf_X[:, 0], axis=-1)
    tf_X2 = tf.expand_dims(tf_X[:, 1], axis=-1)
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

    def sharing_network(tf_projected_sens):
        with tf.variable_scope('sharing_network'):
            tf_inner = tf.layers.conv2d(inputs=tf_projected_sens, filters=64, kernel_size=(5, 5),
                                        strides=(3, 3), activation=tf.nn.relu, padding='valid', name='0',
                                        reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=regularizer)
            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=64, kernel_size=(5, 5), strides=(3, 3),
                                        activation=tf.nn.relu, padding='valid', name='1', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=regularizer)
            tf_inner = tf.layers.batch_normalization(tf_inner, name='2', reuse=tf.AUTO_REUSE)

            tf_inner = tf.layers.max_pooling2d(inputs=tf_inner, pool_size=(3, 3), strides=(3, 3),
                                               padding='valid', name='3')
            tf_inner = tf.layers.batch_normalization(tf_inner, name='4', reuse=tf.AUTO_REUSE)

            tf_inner = tf.layers.flatten(tf_inner, name='5')
            tf_inner = tf.layers.dense(tf_inner, units=1024, name='6', reuse=tf.AUTO_REUSE,
                                       kernel_regularizer=regularizer)
            return tf_inner

    tf_encoding_1 = sharing_network(tf_X1)
    tf_encoding_2 = sharing_network(tf_X2)
    tf_diff = tf.abs(tf.subtract(tf_encoding_1, tf_encoding_2))
    tf_logits = tf.layers.dense(tf_diff, units=1, kernel_regularizer=regularizer)

    logging.info('tf_logits shape: %s', tf_logits.shape)
    return tf_logits


def inference_snn_3(tf_X):
    tf_X1 = tf.expand_dims(tf_X[:, 0], axis=-1)
    tf_X2 = tf.expand_dims(tf_X[:, 1], axis=-1)

    def sharing_network(tf_projected_sens):
        with tf.variable_scope('sharing_network'):

            tf_inner = tf.layers.conv2d(inputs=tf_projected_sens, filters=64, kernel_size=(10, 10),
                                        strides=(1, 1), activation=tf.nn.relu, padding='valid',
                                        reuse=tf.AUTO_REUSE, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_0')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_0')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=128, kernel_size=(7, 7), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_1')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_1')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=128, kernel_size=(4, 4), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_2')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_2')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=256, kernel_size=(4, 4), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_3')
            tf_inner = tf.layers.flatten(tf_inner, name='flatten')

            tf_inner = tf.layers.dense(tf_inner, units=1024, name='dense', reuse=tf.AUTO_REUSE,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3))
            return tf_inner

    tf_encoding_1 = sharing_network(tf_X1)
    tf_encoding_2 = sharing_network(tf_X2)
    tf_diff = tf.abs(tf_encoding_1 - tf_encoding_2)
    tf_logits = tf.squeeze(tf.layers.dense(tf_diff, units=1, activation=tf.nn.sigmoid))

    logging.info('tf_logits: %s', tf_logits)
    return tf_logits


def inference_snn_4(tf_X):
    tf_X1 = tf.expand_dims(tf_X[:, 0], axis=-1)
    tf_X2 = tf.expand_dims(tf_X[:, 1], axis=-1)

    def sharing_network(tf_projected_sens):
        with tf.variable_scope('sharing_network'):

            tf_inner = tf.layers.conv2d(inputs=tf_projected_sens, filters=64, kernel_size=(10, 10),
                                        strides=(1, 1), activation=tf.nn.relu, padding='valid',
                                        reuse=tf.AUTO_REUSE, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_0')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_0')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=128, kernel_size=(7, 7), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_1')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_1')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=128, kernel_size=(4, 4), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_2')
            tf_inner = tf.layers.max_pooling2d(tf_inner, pool_size=(2, 2), strides=(2, 2), padding='valid',
                                               name='max_pool_2')

            tf_inner = tf.layers.conv2d(inputs=tf_inner, filters=256, kernel_size=(4, 4), strides=(1, 1),
                                        activation=tf.nn.relu, padding='valid', reuse=tf.AUTO_REUSE,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4), name='conv_3')
            tf_inner = tf.layers.flatten(tf_inner, name='flatten')

            tf_inner = tf.layers.dense(tf_inner, units=1024, name='dense', reuse=tf.AUTO_REUSE,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3))
            return tf_inner

    tf_encoding_1 = sharing_network(tf_X1)
    tf_encoding_2 = sharing_network(tf_X2)
    tf_diff = tf.abs(tf_encoding_1 - tf_encoding_2)
    tf_logits = tf.squeeze(tf.layers.dense(tf_diff, units=1))

    logging.info('tf_logits: %s', tf_logits)
    return tf_logits


def build_loss_v1(tf_logits, tf_y):
    """

    :param tf_logits: (batch_size, 1)
    :param tf_y: (batch_size, max_time)
    :return:
    """
    tf_logits = tf.squeeze(tf_logits)
    # tf_losses: (batch_size, max_time)
    tf_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=tf_logits)

    tf_aggregated_loss = tf.reduce_mean(tf_losses, axis=-1)
    l2_loss = tf.losses.get_regularization_loss()

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss + l2_loss


def build_loss_v2(tf_logits, tf_y):
    """

    :param tf_logits: (batch_size, 1)
    :param tf_y: (batch_size, max_time)
    :return:
    """
    tf_logits = tf.squeeze(tf_logits)
    # tf_losses: (batch_size, max_time)
    tf_losses = tf.nn.weighted_cross_entropy_with_logits(targets=tf_y, logits=tf_logits, pos_weight=0.2)

    # reduce_mean makes gradient too small, while reduce_sum gives the same meaning without reducing gradient
    tf_aggregated_loss = tf.reduce_mean(tf_losses, axis=-1)

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss


def build_loss_v3(tf_logits, tf_y, reg_weight):
    """
    DEPRECATED
    Loss with regularization factor
    :param tf_logits: (batch_size, 1)
    :param tf_y: (batch_size, max_time)
    :return:
    """
    tf_logits = tf.squeeze(tf_logits)
    # tf_losses: (batch_size, max_time)
    tf_losses = tf.nn.weighted_cross_entropy_with_logits(targets=tf_y, logits=tf_logits, pos_weight=0.2)

    # reduce_mean makes gradient too small, while reduce_sum gives the same meaning without reducing gradient
    tf_aggregated_loss = tf.reduce_mean(tf_losses, axis=-1)
    l2 = reg_weight * tf.add_n(
        [tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name.lower())])

    tf_aggregated_loss += l2

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss


def build_loss_v4(tf_logits, tf_y, reg_weight):
    """
    DEPRECATED
    Loss with regularization factor
    :param tf_logits: (batch_size, 1)
    :param tf_y: (batch_size, max_time)
    :return:
    """
    tf_logits = tf.squeeze(tf_logits)
    # tf_losses: (batch_size, max_time)
    tf_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=tf_logits)

    # reduce_mean makes gradient too small, while reduce_sum gives the same meaning without reducing gradient
    tf_aggregated_loss = tf.reduce_mean(tf_losses, axis=-1)
    l2 = reg_weight * tf.add_n(
        [tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name.lower())])

    tf_aggregated_loss += l2

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss


def build_loss_snn_1(tf_dis, tf_y):
    tmp = tf_y * tf_dis
    tmp2 = (1 - tf_y) * tf.maximum((1 - tf_dis), 0)

    return tf.reduce_mean(tmp + tmp2)


def build_optimize_v1(tf_loss, learning_rate=0.05):
    """
    Return tensor optimizer and global step
    """
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer())
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss, global_step=tf_global_step)
    return optimizer, tf_global_step


def build_predict_prob(tf_logit):
    return tf.nn.sigmoid(tf_logit, name='tf_prediction')


def build_predict(tf_prob, threshold=0.5):
    """
    Convert from tensor logit to tensor one hot
    """
    prediction = tf.cast(tf_prob, tf.float64)
    threshold = float(threshold)
    return tf.cast(tf.greater(prediction, threshold), tf.float32)


def build_predict_2(tf_prob, threshold=0.5):
    """
    Convert from tensor logit to tensor one hot
    """
    prediction = tf.cast(tf_prob, tf.float64)
    threshold = float(threshold)
    return tf.cast(tf.less_equal(prediction, threshold), tf.float32)


def build_precision_class(tf_predict, tf_y, class_value):
    true_positive = tf.reduce_sum(
        tf.boolean_mask(tf.cast(tf.equal(tf_predict, tf_y), tf.float32), mask=tf.equal(tf_predict, class_value)))
    false_positive = tf.reduce_sum(
        tf.boolean_mask(tf.cast(tf.not_equal(tf_predict, tf_y), tf.float32), mask=tf.equal(tf_predict, class_value)))
    c = true_positive / (true_positive + false_positive)
    c = tf.cond(tf.equal(tf.reduce_sum(tf.cast(tf.equal(tf_predict, class_value), tf.float32)), 0), lambda: 1.,
                lambda: c)
    return c


def build_recall_class(tf_pred, tf_y, class_value):
    true_positive = tf.reduce_sum(
        tf.boolean_mask(tf.cast(tf.equal(tf_pred, tf_y), tf.float32), mask=tf.equal(tf_pred, class_value)))
    false_negative = tf.reduce_sum(
        tf.boolean_mask(tf.cast(tf.not_equal(tf_pred, tf_y), tf.float32), mask=tf.not_equal(tf_pred, class_value)))
    c = true_positive / (true_positive + false_negative)
    c = tf.cond(tf.equal(tf.reduce_sum(tf.cast(tf.equal(tf_y, class_value), tf.float32)), 0), lambda: 1.,
                lambda: c)
    return c
