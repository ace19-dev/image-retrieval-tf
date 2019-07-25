from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import numpy as np
import tensorflow as tf

import model
import train_data
import val_data
from utils import train_utils, aug_utils

# from slim.nets import inception_v4

slim = tf.contrib.slim


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_logdir', './tfmodels',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('ckpt_name_to_save', 'resnet_v2_50.ckpt',
                    'Name to save checkpoint file')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './tfmodels/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', 0.05,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 1e-4,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .2000,
                   'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_float('training_number_of_steps', 300000,
                   'The number of steps used for training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')
flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 0.002,
                   'Learning rate employed during slow start.')

# Settings for fine-tuning the network.
flags.DEFINE_string('saved_checkpoint_dir',
                    # './tfmodels',
                    None,
                    'Saved checkpoint dir.')
flags.DEFINE_string('pre_trained_checkpoint',
                    'pre-trained/resnet_v2_50.ckpt',
                    # None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    'resnet_v2_50/logits,resnet_v2_50/SpatialSqueeze,resnet_v2_50/predictions',
                    # None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    # 'resnet_v2_50/logits,resnet_v2_50/SpatialSqueeze,resnet_v2_50/predictions',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'resnet_v2_50',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/v2-plant-seedlings-dataset-resized',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 80,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('val_batch_size', 32, 'validation batch size')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'Black_grass,Charlock,Cleavers,Common_Chickweed,Common_wheat,Fat_Hen,'
                    'Loose_Silky_bent,Maize,Scentless_Mayweed,Shepherds_Purse,'
                    'Small_flowered_Cranesbill,Sugar_beet',
                    'Labels to use')

# temporary constant
TRAIN_DATA_SIZE = 263+384+285+606+215+457+648+219+516+233+490+393   # 4709
VALIDATE_DATA_SIZE = 46+68+50+107+38+81+114+38+91+41+86+70     # 830



def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    tf.compat.v1.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        X = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, 3], name='X')
        ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        keep_prob = tf.compat.v1.placeholder(tf.float32, [], name='keep_prob')
        # learning_rate = tf.placeholder(tf.float32, [])

        # apply SENet
        logits, end_points = model.basic_model(X,
                                               num_classes=num_classes,
                                               is_training=is_training,
                                               keep_prob=keep_prob,
                                               attention_module='se_block')

        # logits, features = model.deep_cosine_metric_learning(X,
        #                                                      num_classes=num_classes,
        #                                                      is_training=is_training,
        #                                                      keep_prob=keep_prob,
        #                                                      attention_module='se_block')


        # Print name and shape of each tensor.
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        tf.logging.info("Layers")
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        for k, v in end_points.items():
            tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        # # Print name and shape of parameter nodes  (values not yet initialized)
        # tf.logging.info("++++++++++++++++++++++++++++++++++")
        # tf.logging.info("Parameters")
        # tf.logging.info("++++++++++++++++++++++++++++++++++")
        # for v in slim.get_model_variables():
        #     tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        prediction = tf.argmax(logits, axis=1, name='prediction')
        correct_prediction = tf.equal(prediction, ground_truth)
        confusion_matrix = tf.math.confusion_matrix(ground_truth,
                                                    prediction,
                                                    num_classes=num_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        # summaries.add(tf.compat.v1.summary.scalar('accuracy', accuracy))

        # Define loss
        tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=ground_truth,
                                                         logits=logits)

        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
        summaries.add(tf.compat.v1.summary.scalar('accuracy', accuracy))

        # Add summaries for model variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.compat.v1.summary.histogram(variable.op.name, variable))

        # Add summaries for losses.
        for loss in tf.compat.v1.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.compat.v1.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.compat.v1.check_numerics(total_loss, 'Loss is inf or nan.')
        summaries.add(tf.compat.v1.summary.scalar('total_loss', total_loss))

        # TensorBoard: How to plot histogram for gradients
        # grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')


        ###############
        # Prepare data
        ###############
        # training dateset
        tfrecord_filenames = tf.compat.v1.placeholder(tf.string, shape=[])
        tr_dataset = train_data.Dataset(tfrecord_filenames,
                                        FLAGS.batch_size,
                                        FLAGS.how_many_training_epochs,
                                        FLAGS.height,
                                        FLAGS.width)
        iterator = tr_dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        # validation dateset
        val_dataset = val_data.Dataset(tfrecord_filenames,
                                       FLAGS.val_batch_size,
                                       FLAGS.how_many_training_epochs,
                                       FLAGS.height,
                                       FLAGS.width)
        val_iterator = val_dataset.dataset.make_initializable_iterator()
        val_next_batch = val_iterator.get_next()

        sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        with tf.compat.v1.Session(config = sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            # Add the summaries. These contain the summaries
            # created by model and either optimize() or _gather_loss().
            summaries |= set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

            # Merge all summaries together.
            summary_op = tf.compat.v1.summary.merge(list(summaries))
            train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir, graph)
            validation_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/validation', graph)

            # Create a saver object which will save all the variables
            saver = tf.compat.v1.train.Saver()
            if FLAGS.saved_checkpoint_dir:
                if tf.gfile.IsDirectory(FLAGS.train_logdir):
                    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_logdir)
                else:
                    checkpoint_path = FLAGS.train_logdir
                saver.restore(sess, checkpoint_path)

            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            tr_batches = int(TRAIN_DATA_SIZE / FLAGS.batch_size)
            if TRAIN_DATA_SIZE % FLAGS.batch_size > 0:
                tr_batches += 1
            val_batches = int(VALIDATE_DATA_SIZE / FLAGS.val_batch_size)
            if VALIDATE_DATA_SIZE % FLAGS.val_batch_size > 0:
                val_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either be a string,
            # a list of strings, or a tf.Tensor of strings.
            train_record_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            validate_record_filenames = os.path.join(FLAGS.dataset_dir, 'validate.record')

            ############################
            # Training loop.
            ############################
            for num_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("------------------------------------")
                print(" Epoch {} ".format(num_epoch))
                print("------------------------------------")

                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: train_record_filenames})
                for step in range(tr_batches):
                    filenames, train_batch_xs, train_batch_ys = sess.run(next_batch)
                    # # Verify image
                    # # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = train_batch_xs.shape[0]
                    # # n_view = train_batch_xs.shape[1]
                    # for i in range(n_batch):
                    #     img = train_batch_xs[i]
                    #     # scipy.misc.toimage(img).show() Or
                    #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
                    #     # cv2.imshow(str(train_batch_ys[idx]), img)
                    #     cv2.waitKey(100)
                    #     cv2.destroyAllWindows()

                    # augmented_batch_xs = aug_utils.aug(train_batch_xs)
                    # # Verify image
                    # # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = augmented_batch_xs.shape[0]
                    # # n_view = train_batch_xs.shape[1]
                    # for i in range(n_batch):
                    #     img = augmented_batch_xs[i]
                    #     # scipy.misc.toimage(img).show() Or
                    #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite('/home/ace19/Pictures/aug/' + str(i) + '.png', img)
                    #     # cv2.imshow(str(train_batch_ys[idx]), img)
                    #     cv2.waitKey(100)
                    #     cv2.destroyAllWindows()

                    # Run the graph with this batch of training data and learning rate policy.
                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.run([learning_rate, summary_op, accuracy, total_loss, train_op],
                                 feed_dict={
                                     X: train_batch_xs,
                                     ground_truth: train_batch_ys,
                                     is_training: True,
                                     keep_prob: 0.8
                                 })
                    train_writer.add_summary(train_summary, num_epoch)
                    # train_writer.add_summary(grad_vals, num_epoch)
                    tf.logging.info('Epoch #%d, Step #%d, rate %.10f, accuracy %.1f%%, loss %f' %
                                    (num_epoch, step, lr, train_accuracy * 100, train_loss))

                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.logging.info('--------------------------')
                tf.logging.info(' Start validation ')
                tf.logging.info('--------------------------')

                total_val_accuracy = 0
                validation_count = 0
                total_conf_matrix = None
                sess.run(val_iterator.initializer, feed_dict={tfrecord_filenames: validate_record_filenames})
                for step in range(val_batches):
                    filenames, validation_batch_xs, validation_batch_ys = sess.run(val_next_batch)
                    # random augmentation for TTA
                    # augmented_val_batch_xs = aug_utils.aug(validation_batch_xs)
                    # # Verify image
                    # # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = validation_batch_xs.shape[0]
                    # # n_view = train_batch_xs.shape[1]
                    # for i in range(n_batch):
                    #     img = validation_batch_xs[i]
                    #     # scipy.misc.toimage(img).show() Or
                    #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
                    #     # cv2.imshow(str(train_batch_ys[idx]), img)
                    #     cv2.waitKey(100)
                    #     cv2.destroyAllWindows()

                    val_summary, val_accuracy, conf_matrix = sess.run(
                        [summary_op, accuracy, confusion_matrix],
                        feed_dict={
                            X: validation_batch_xs,
                            ground_truth: validation_batch_ys,
                            is_training: False,
                            keep_prob: 1.0
                        })
                    validation_writer.add_summary(val_summary, num_epoch)

                    total_val_accuracy += val_accuracy
                    validation_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                total_val_accuracy /= validation_count
                tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.logging.info('Validation accuracy = %.1f%% (N=%d)' %
                                (total_val_accuracy * 100, VALIDATE_DATA_SIZE))

                # Save the model checkpoint periodically.
                if (num_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, num_epoch)
                    saver.save(sess, checkpoint_path, global_step=num_epoch)


if __name__ == '__main__':
    if tf.io.gfile.exists(FLAGS.train_logdir):
        tf.io.gfile.rmtree(FLAGS.train_logdir)
    tf.io.gfile.makedirs(FLAGS.train_logdir)

    tf.compat.v1.app.run()
