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
                                   attention_module=attention_module,
                                   scope='resnet_v2_50')

    net = slim.batch_norm(net, scope='batch_norm')
    end_points['batch_norm'] = net
    net = slim.flatten(net, scope='flatten')
    end_points['flatten'] = net
    net = slim.fully_connected(net, 256, scope='fc1')
    end_points['fc1'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc2')
    end_points['fc2'] = logits

    return logits, end_points


def deep_cosine_metric_learning(inputs,
                                num_classes,
                                is_training=True,
                                keep_prob=0.8,
                                attention_module=None,
                                scope=''):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = \
            resnet_v2.resnet_v2_50(inputs,
                                   num_classes=num_classes,
                                   is_training=is_training,
                                   attention_module=attention_module,
                                   scope='resnet_v2_50')

        ###############################
        # deep cosine metric learning
        ###############################
        # (?,7,7,2048)
        feature_dim = net.get_shape().as_list()[-1]
        net = slim.flatten(net)
        net = slim.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net,
                                   feature_dim,
                                   normalizer_fn=slim.batch_norm,
                                   weights_regularizer=slim.l2_regularizer(1e-8),
                                   scope='fc1')

        features = net

        # Features in rows, normalize axis 1.
        # The final l2 normalization projects features onto the unit hypersphere
        # for application of the cosine softmax classifier.
        features = tf.nn.l2_normalize(features, axis=1)

        with tf.compat.v1.variable_scope("ball"):
            weights = \
                slim.model_variable("mean_vectors",
                                    (feature_dim, int(num_classes)),
                                    initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                    regularizer=None)
            # The scaling parameter κ controls
            # the shape of the conditional class probabilities
            scale = \
                slim.model_variable("scale",
                                    (),
                                    tf.float32,
                                    initializer=tf.constant_initializer(0., tf.float32),
                                    regularizer=slim.l2_regularizer(1e-1))

            tf.compat.v1.summary.scalar("scale", scale)
            scale = tf.nn.softplus(scale)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = tf.nn.l2_normalize(weights, axis=0)
        logits = scale * tf.matmul(features, weights_normed)

    return logits, features  # use it for retrieval.
