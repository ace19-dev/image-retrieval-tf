from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_v2

slim = tf.contrib.slim

batch_norm_params = {
    'decay': 0.997,  # batch_norm_decay
    'epsilon': 1e-5,  # batch_norm_epsilon
    'scale': True,  # batch_norm_scale
    'updates_collections': tf.compat.v1.GraphKeys.UPDATE_OPS,  # batch_norm_updates_collections
    'is_training': True,  # is_training
    'fused': None,  # Use fused batch norm if possible.
}


def basic_model(inputs,
                num_classes,
                is_training=True,
                is_resue=False,
                keep_prob=0.8,
                attention_module=None,
                scope='basic_model'):
    '''
    :param inputs: N x H x W x C tensor
    :return:
    '''

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = \
            resnet_v2.resnet_v2_50(inputs,
                                   num_classes=num_classes,
                                   is_training=is_training,
                                   reuse=is_resue,
                                   attention_module=attention_module,
                                   scope='resnet_v2_50')

    #     # Global average pooling.
    #     net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
    #     end_points['global_pool'] = net
    #
    # batch_norm_params['is_training'] = is_training
    # # net = slim.batch_norm(net, scope='batch_norm')
    # # end_points['batch_norm'] = net
    # net = slim.flatten(net, scope='flatten')
    # end_points['flatten'] = net
    # net = slim.fully_connected(net, 256, normalizer_fn=slim.batch_norm,
    #                            normalizer_params=batch_norm_params, scope='fc1')
    # end_points['fc1'] = net
    #
    # net = slim.fully_connected(net, num_classes, normalizer_fn=slim.batch_norm,
    #                            normalizer_params=batch_norm_params, activation_fn=None, scope='fc2')
    # end_points['fc2'] = net

    logits = net

    return logits, end_points


def deep_cosine_softmax(inputs,
                        num_classes,
                        is_training=True,
                        keep_prob=0.8,
                        attention_module=None,
                        scope=''):

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, _ = \
            resnet_v2.resnet_v2_50(inputs,
                                   num_classes=num_classes,
                                   is_training=is_training,
                                   attention_module=attention_module,
                                   scope='resnet_v2_50')

    # ##############################
    # cosine-softmax
    # ##############################
    # (?,7,7,2048)
    feature_dim = net.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    net = slim.flatten(net)

    net = slim.dropout(net, keep_prob=keep_prob)
    net = slim.fully_connected(
        net, feature_dim, normalizer_fn=batch_norm_fn,
        weights_regularizer=slim.l2_regularizer(1e-8),
        scope="fc1", weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        biases_initializer=tf.zeros_initializer())

    features = net

    # TODO: check shape
    # Features in rows, normalize axis 1.
    features = tf.nn.l2_normalize(features, dim=1)

    with slim.variable_scope.variable_scope("ball", reuse=None):
        weights = slim.model_variable(
            "mean_vectors", (feature_dim, int(num_classes)),
            initializer=tf.truncated_normal_initializer(stddev=1e-3),
            regularizer=None)
        scale = slim.model_variable(
            "scale", (), tf.float32,
            initializer=tf.constant_initializer(0., tf.float32),
            regularizer=slim.l2_regularizer(1e-1))
        tf.summary.scalar("scale", scale)
        scale = tf.nn.softplus(scale)

    # Mean vectors in colums, normalize axis 0.
    weights_normed = tf.nn.l2_normalize(weights, dim=0)
    logits = scale * tf.matmul(features, weights_normed)

    return logits, features  # use it for retrieval.
