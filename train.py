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
from utils import train_utils, aug_utils, train_helper
from checkmate import BestCheckpointSaver, get_best_checkpoint

# from slim.nets import inception_v4

slim = tf.contrib.slim


flags = tf.app.flags
FLAGS = flags.FLAGS

# Multi GPU
flags.DEFINE_integer('num_gpu', 2, 'number of GPU')

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
flags.DEFINE_float('base_learning_rate', 0.0002,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 1e-4,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .500,
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
flags.DEFINE_integer('slow_start_step', 210,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 0.00002,
                   'Learning rate employed during slow start.')

# Settings for fine-tuning the network.
flags.DEFINE_string('saved_checkpoint_dir',
                    # './tfmodels',
                    None,
                    'Saved checkpoint dir.')
# flags.DEFINE_string('saved_checkpoint_path',
#                     # './tfmodels/best_resnet_v2_50.ckpt',
#                     None,
#                     'Saved checkpoint path.')
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

flags.DEFINE_integer('how_many_training_epochs', 100,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('val_batch_size', 128, 'validation batch size')
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

TEN_CROP = 10


def show_batch_data(filenames, batch_x, batch_y, additional_path=None):
    default_path = '/home/ace19/Pictures/'
    if additional_path is not None:
        default_path = os.path.join(default_path, additional_path)
        if not os.path.exists(default_path):
            os.makedirs(default_path)

    assert not np.any(np.isnan(batch_x))

    n_batch = batch_x.shape[0]
    # n_view = batch_x.shape[1]
    for i in range(n_batch):
        img = batch_x[i]
        # scipy.misc.toimage(img).show() Or
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(default_path, str(i) + '.png'), img)
        # cv2.imshow(str(batch_y[idx]), img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)


    with tf.Graph().as_default() as graph:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        X = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, 3], name='X')
        ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        keep_prob = tf.compat.v1.placeholder(tf.float32, [], name='keep_prob')
        tfrecord_filenames = tf.compat.v1.placeholder(tf.string, shape=[])

        # # Print name and shape of each tensor.
        # tf.compat.v1.logging.info("++++++++++++++++++++++++++++++++++")
        # tf.compat.v1.logging.info("Layers")
        # tf.compat.v1.logging.info("++++++++++++++++++++++++++++++++++")
        # for k, v in end_points.items():
        #     tf.compat.v1.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))
        #
        # # # Print name and shape of parameter nodes  (values not yet initialized)
        # # tf.compat.v1.logging.info("++++++++++++++++++++++++++++++++++")
        # # tf.compat.v1.logging.info("Parameters")
        # # tf.compat.v1.logging.info("++++++++++++++++++++++++++++++++++")
        # # for v in slim.get_model_variables():
        # #     tf.compat.v1.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
        # # Add summaries for model variables.
        # for variable in slim.get_model_variables():
        #     summaries.add(tf.compat.v1.summary.histogram(variable.op.name, variable))
        #
        # # Add summaries for losses.
        # for loss in tf.compat.v1.get_collection(tf.GraphKeys.LOSSES):
        #     summaries.add(tf.compat.v1.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

        optimizers = \
            [tf.compat.v1.train.MomentumOptimizer(learning_rate, FLAGS.momentum) for _ in range(FLAGS.num_gpu)]

        logits = []
        losses = []
        grad_list = []
        filename_batch = []
        image_batch = []
        gt_batch = []
        for gpu_idx in range(FLAGS.num_gpu):
            tf.compat.v1.logging.info('creating gpu tower @ %d' % (gpu_idx + 1))
            image_batch.append(X)
            gt_batch.append(ground_truth)

            scope_name = 'tower%d' % gpu_idx
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope(scope_name):
                # apply SENet
                logit, _ = model.basic_model(X,
                                             num_classes=num_classes,
                                             is_training=is_training,
                                             is_reuse=False,
                                             keep_prob=keep_prob,
                                             attention_module='se_block')
                # logits, features = model.deep_cosine_metric_learning(X,
                #                                                      num_classes=num_classes,
                #                                                      is_training=is_training,
                #                                                      keep_prob=keep_prob,
                #                                                      attention_module='se_block')
                logit = tf.cond(is_training,
                                lambda: tf.identity(logit),
                                lambda: tf.reduce_mean(tf.reshape(logit, [FLAGS.val_batch_size, TEN_CROP, -1]), axis=1))
                logits.append(logit)

                l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth,
                                                                   logits=logit)
                losses.append(l)
                loss_w_reg = tf.reduce_sum(l) + tf.add_n(slim.losses.get_regularization_losses(scope=scope_name))

                grad_list.append(
                    [x for x in optimizers[gpu_idx].compute_gradients(loss_w_reg) if x[0] is not None])

        y_hat = tf.concat(logits, axis=0)
        image_batch = tf.concat(image_batch, axis=0)
        gt_batch = tf.concat(gt_batch, axis=0)

        # loss
        top1_acc = tf.reduce_mean(
            tf.cast(tf.nn.in_top_k(y_hat, gt_batch, k=1), dtype=tf.float32)
        )
        summaries.add(tf.compat.v1.summary.scalar('top1_acc', top1_acc))
        # top5_acc = tf.reduce_mean(
        #     tf.cast(tf.nn.in_top_k(y_hat, gt_batch, k=5), dtype=tf.float32)
        # )
        # summaries.add(tf.compat.v1.summary.scalar('top5_acc', top5_acc))
        prediction = tf.argmax(y_hat, axis=1, name='prediction')
        confusion_matrix = tf.math.confusion_matrix(gt_batch,
                                                    prediction,
                                                    num_classes=num_classes)

        loss = tf.reduce_mean(losses)
        loss = tf.compat.v1.check_numerics(loss, 'Loss is inf or nan.')
        summaries.add(tf.compat.v1.summary.scalar('loss', loss))

        # use NCCL
        grads, all_vars = train_helper.split_grad_list(grad_list)
        reduced_grad = train_helper.allreduce_grads(grads, average=True)
        grads = train_helper.merge_grad_list(reduced_grad, all_vars)

        # optimizer using NCCL
        train_ops = []
        for idx, grad_and_vars in enumerate(grads):
            # apply_gradients may create variables. Make them LOCAL_VARIABLESZ¸¸¸¸¸¸
            with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='tower%d' % idx)
                with tf.control_dependencies(update_ops):
                    train_ops.append(
                        optimizers[idx].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx), global_step=global_step)
                    )
                # TODO:
                # TensorBoard: How to plot histogram for gradients
                # grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])

        optimize_op = tf.group(*train_ops, name='train_op')

        sync_op = train_helper.get_post_init_ops()

        # Create a saver object which will save all the variables
        saver = tf.compat.v1.train.Saver()
        best_ckpt_saver = BestCheckpointSaver(
            save_dir=FLAGS.train_logdir,
            num_to_keep=100,
            maximize=False,
            saver=saver
        )
        best_val_loss = 99999
        best_val_acc = 0

        start_epoch = 0
        epoch_count = tf.Variable(start_epoch, trainable=False)
        epoch_count_add = tf.assign(epoch_count, epoch_count + 1)

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

            if FLAGS.saved_checkpoint_dir:
                if tf.gfile.IsDirectory(FLAGS.saved_checkpoint_dir):
                    checkpoint_path = tf.train.latest_checkpoint(FLAGS.saved_checkpoint_dir)
                else:
                    checkpoint_path = FLAGS.saved_checkpoint_dir
                saver.restore(sess, checkpoint_path)

            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            sess.run(sync_op)


            ###############
            # Prepare data
            ###############
            # training dateset
            tr_dataset = train_data.Dataset(tfrecord_filenames,
                                            FLAGS.batch_size // FLAGS.num_gpu,
                                            num_classes,
                                            FLAGS.how_many_training_epochs,
                                            TRAIN_DATA_SIZE,
                                            FLAGS.height,
                                            FLAGS.width)
            iterator = tr_dataset.dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            # validation dateset
            val_dataset = val_data.Dataset(tfrecord_filenames,
                                           FLAGS.val_batch_size // FLAGS.num_gpu,
                                           num_classes,
                                           FLAGS.how_many_training_epochs,
                                           VALIDATE_DATA_SIZE,
                                           FLAGS.height,
                                           FLAGS.width)
            val_iterator = val_dataset.dataset.make_initializable_iterator()
            val_next_batch = val_iterator.get_next()

            # Get the number of training/validation steps per epoch
            tr_batches = int(TRAIN_DATA_SIZE / (FLAGS.batch_size // FLAGS.num_gpu))
            if TRAIN_DATA_SIZE % (FLAGS.batch_size // FLAGS.num_gpu) > 0:
                tr_batches += 1
            val_batches = int(VALIDATE_DATA_SIZE / (FLAGS.val_batch_size // FLAGS.num_gpu))
            if VALIDATE_DATA_SIZE % (FLAGS.val_batch_size // FLAGS.num_gpu) > 0:
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

                sess.run(epoch_count_add)
                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: train_record_filenames})
                for step in range(tr_batches):
                    filenames, train_batch_xs, train_batch_ys = sess.run(next_batch)
                    # show_batch_data(filenames, train_batch_xs, train_batch_ys)
                    #
                    # augmented_batch_xs = aug_utils.aug(train_batch_xs)
                    # show_batch_data(filenames, augmented_batch_xs,
                    #                 train_batch_ys, 'aug')

                    # Run the graph with this batch of training data and learning rate policy.
                    lr, train_summary, train_top1_acc, train_loss, _ = \
                        sess.run([learning_rate, summary_op, top1_acc, loss, optimize_op],
                                 feed_dict={
                                     X: train_batch_xs,
                                     ground_truth: train_batch_ys,
                                     is_training: True,
                                     keep_prob: 0.7
                                 })
                    train_writer.add_summary(train_summary, num_epoch)
                    # train_writer.add_summary(grad_vals, num_epoch)
                    tf.compat.v1.logging.info('Epoch #%d, Step #%d, rate %.6f, top1_acc=%.3f, loss %.5f' %
                                    (num_epoch, step, lr, train_top1_acc, train_loss))

                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.compat.v1.logging.info('--------------------------')
                tf.compat.v1.logging.info(' Start validation ')
                tf.compat.v1.logging.info('--------------------------')

                total_val_losses = 0.0
                total_val_top1_acc = 0.0
                # total_val_accuracy = 0
                val_count = 0
                total_conf_matrix = None

                sess.run(val_iterator.initializer, feed_dict={tfrecord_filenames: validate_record_filenames})
                for step in range(val_batches):
                    filenames, validation_batch_xs, validation_batch_ys = sess.run(val_next_batch)
                    # TTA
                    batch_size, n_crops, c, h, w = validation_batch_xs.shape
                    # fuse batch size and ncrops
                    tencrop_val_batch_xs = np.reshape(validation_batch_xs, (-1, c, h, w))
                    # show_batch_data(filenames, tencrop_val_batch_xs, validation_batch_ys)

                    # augmented_val_batch_xs = aug_utils.aug(tencrop_val_batch_xs)
                    # show_batch_data(filenames, augmented_val_batch_xs,
                    #                 validation_batch_ys, 'aug')

                    # TODO: Verify TTA(TenCrop)
                    val_summary, val_loss, val_top1_acc, _confusion_matrix = sess.run(
                        [summary_op, loss, top1_acc, confusion_matrix],
                        feed_dict={
                            X: tencrop_val_batch_xs,
                            ground_truth: validation_batch_ys,
                            is_training: False,
                            keep_prob: 1.0
                        })
                    validation_writer.add_summary(val_summary, num_epoch)

                    total_val_losses += val_loss
                    total_val_top1_acc += val_top1_acc

                    # total_val_accuracy += val_top1_acc
                    val_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = _confusion_matrix
                    else:
                        total_conf_matrix += _confusion_matrix

                total_val_losses /= val_count
                total_val_top1_acc /= val_count

                # total_val_accuracy /= val_count
                tf.compat.v1.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
                tf.compat.v1.logging.info('Validation loss = %.5f' % total_val_losses)
                tf.compat.v1.logging.info('Validation top1 accuracy = %.3f%% (N=%d)' %
                                (total_val_top1_acc, VALIDATE_DATA_SIZE))

                # Save the model checkpoint periodically.
                if (num_epoch <= FLAGS.how_many_training_epochs-1):
                    # best_checkpoint_path = os.path.join(FLAGS.train_logdir, 'best_' + FLAGS.ckpt_name_to_save)
                    # tf.compat.v1.logging.info('Saving to "%s"', best_checkpoint_path)
                    # saver.save(sess, best_checkpoint_path, global_step=num_epoch)

                    # save & keep best model wrt. validation loss
                    best_ckpt_saver.handle(total_val_losses, sess, epoch_count)

                    if best_val_loss > total_val_losses:
                        best_val_loss = total_val_losses
                        best_val_acc = total_val_top1_acc

                    # periodic synchronization
                    sess.run(sync_op)

            chk_path = get_best_checkpoint(FLAGS.train_logdir, select_maximum_value=False)
            tf.compat.v1.logging.info('training done. best_model val_loss=%.5f, top1_acc=%.3f, ckpt=%s' % (
                best_val_loss, best_val_acc, chk_path))


if __name__ == '__main__':
    tf.compat.v1.logging.info('Creating train logdir: %s', FLAGS.train_logdir)
    tf.io.gfile.makedirs(FLAGS.train_logdir)

    tf.compat.v1.app.run()
