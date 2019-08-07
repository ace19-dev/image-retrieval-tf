import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from retrieval import retrieval_data, matching
from utils import aug_utils
import model
import train_data


slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/materials',
                    'Where the dataset reside.')

flags.DEFINE_string('output_dir',
                    '/home/ace19/dl_results/image_retrieve/_result',
                    'Where the dataset reside.')

flags.DEFINE_string('checkpoint_path',
                    '../tfmodels',
                    'Directory where to read training checkpoints.')

flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
# flags.DEFINE_string('labels',
#                     'airplane,bed,bookshelf,toilet,vase',
#                     'number of classes')

# # retrieval params
# flags.DEFINE_float('max_cosine_distance', 0.2,
#                    'Gating threshold for cosine distance')
# flags.DEFINE_string('nn_budget', None,
#                     'Maximum size of the appearance descriptors gallery. '
#                     'If None, no budget is enforced.')


MODELNET_GALLERY_SIZE = 43955
MODELNET_QUERY_SIZE = 300

TOP_N = 5
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


def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r,c]])
        distances.append(col)

    return distances


def match_n(top_n, galleries, queries):
    # The distance metric used for measurement to query.
    metric = matching.NearestNeighborDistanceMetric("cosine")
    distance_matrix = metric.distance(queries, galleries)

    # top_indice = np.argmin(distance_matrix, axis=1)
    # top_n_indice = np.argpartition(distance_matrix, top_n, axis=1)[:, :top_n]
    # top_n_dist = _print_distances(distance_matrix, top_n_indice)
    # top_n_indice2 = np.argsort(top_n_dist, axis=1)
    # dist2 = _print_distances(distance_matrix, top_n_indice2)

    # TODO: need improvement.
    top_n_indice = np.argsort(distance_matrix, axis=1)[:, :top_n]
    top_n_distance = _print_distances(distance_matrix, top_n_indice)

    return top_n_indice, top_n_distance


def show_retrieval_result(top_n_indice, top_n_distance, gallery_path_list, query_path_list):
    col = top_n_indice.shape[1]
    for row_idx, query_img_path in enumerate(query_path_list):
        fig, axes = plt.subplots(ncols=6, figsize=(15, 4))
        # fig.suptitle(query_img_path.split('/')[-1], fontsize=12, fontweight='bold')
        axes[0].set_title(query_img_path.split('/')[-1], color='r', fontweight='bold')
        axes[0].imshow(Image.open(query_img_path))

        for i in range(col):
            img_path = gallery_path_list[top_n_indice[row_idx, i]]
            axes[i+1].set_title(img_path.split('/')[-1])
            axes[i+1].imshow(Image.open(img_path))
        # plt.show()
        print(" Retrieval result {} create.".format(row_idx+1))
        fig.savefig(os.path.join(FLAGS.output_dir, query_img_path.split('/')[-1]))
        plt.close()


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.compat.v1.placeholder(tf.float32,
                                 [None, FLAGS.height, FLAGS.width, 3],
                                 name='X')
    is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
    keep_prob = tf.compat.v1.placeholder(tf.float32, [], name='keep_prob')

    features, _ = model.deep_cosine_softmax(X,
                                            num_classes=num_classes,
                                            is_training=is_training,
                                            is_reuse=False,
                                            keep_prob=keep_prob,
                                            attention_module='se_block')
    features = tf.cond(is_training,
                       lambda: tf.identity(features),
                       lambda: tf.reduce_mean(tf.reshape(features, [FLAGS.batch_size, TEN_CROP, -1]), axis=1))

    # Prepare query source data
    tfrecord_filenames = tf.placeholder(tf.string, shape=[])
    train_dataset = train_data.Dataset(tfrecord_filenames,
                                       FLAGS.batch_size,
                                       num_classes,
                                       None,
                                       MODELNET_GALLERY_SIZE,
                                       FLAGS.height,
                                       FLAGS.width)
    train_iterator = train_dataset.dataset.make_initializable_iterator()
    train_next_batch = train_iterator.get_next()

    _dataset = retrieval_data.Dataset(tfrecord_filenames,
                                      FLAGS.batch_size,
                                      num_classes,
                                      None,
                                      MODELNET_QUERY_SIZE,
                                      256,  # 256 ~ 480
                                      256)
    iterator = _dataset.dataset.make_initializable_iterator()
    next_batch = iterator.get_next()



    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        saver.restore(sess, checkpoint_path)

        # global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches_gallery = int(MODELNET_GALLERY_SIZE / FLAGS.batch_size)
        if MODELNET_GALLERY_SIZE % FLAGS.batch_size > 0:
            batches_gallery += 1
        batches_query = int(MODELNET_QUERY_SIZE / FLAGS.batch_size)
        if MODELNET_QUERY_SIZE % FLAGS.batch_size > 0:
            batches_query += 1

        gallery_tf_filenames = os.path.join(FLAGS.dataset_dir, 'gallery.record')
        query_tf_filenames = os.path.join(FLAGS.dataset_dir, 'query.record')

        # TODO: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # TODO: It is better to create encode func which replace below codes
        # TODO: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        gallery_features_list = []
        gallery_path_list = []
        sess.run(train_iterator.initializer, feed_dict={tfrecord_filenames: gallery_tf_filenames})
        for i in range(batches_gallery):
            filenames, gallery_batch_xs, gallery_batch_ys = sess.run(train_next_batch)
            # show_batch_data(filenames, gallery_batch_xs, gallery_batch_ys)

            # (10,512)
            _f = sess.run(features, feed_dict={X: gallery_batch_xs,
                                               is_training:True,
                                               keep_prob: 1.0})
            gallery_features_list.extend(_f)
            gallery_path_list.extend(filenames)

        # query images
        query_features_list = []
        query_path_list = []
        sess.run(iterator.initializer, feed_dict={tfrecord_filenames: query_tf_filenames})
        for i in range(batches_query):
            filenames, query_batch_xs, query_batch_ys  = sess.run(next_batch)
            # show_batch_data(filenames, query_batch_xs, query_batch_ys)

            # TTA
            batch_size, n_crops, c, h, w = query_batch_xs.shape
            # fuse batch size and ncrops
            tencrop_query_batch_xs = np.reshape(query_batch_xs, (-1, c, h, w))

            # (10,512)
            _f = sess.run(features, feed_dict={X: tencrop_query_batch_xs,
                                               is_training:False,
                                               keep_prob: 1.0})
            query_features_list.extend(_f)
            query_path_list.extend(filenames)

        if len(query_features_list) == 0:
            print('No query data!!')
            return

        # matching
        top_n_indice, top_n_distance = \
            match_n(TOP_N, tf.stack(gallery_features_list), tf.stack(query_features_list))

        # Show n images from the gallery similar to the query image.
        show_retrieval_result(top_n_indice, top_n_distance, gallery_path_list, query_path_list)


if __name__ == '__main__':
    tf.compat.v1.app.run()