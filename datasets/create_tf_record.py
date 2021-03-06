from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import random

from PIL import Image, ImageStat
import tensorflow as tf

from datasets import dataset_utils


flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/materials',
                    'Root Directory to dataset.')
flags.DEFINE_string('output_path',
                    '/home/ace19/dl_data/materials/query.record',
                    'Path to output TFRecord')
flags.DEFINE_string('dataset_category',
                    'query',
                    'dataset category, train|validation|test')

FLAGS = flags.FLAGS


def get_label_map(label_to_index):
    label_map = {}
    # cls_lst = os.listdir(FLAGS.dataset_dir)
    cls_path = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_category)
    cls_lst = os.listdir(cls_path)
    for i, cls in enumerate(cls_lst):
        data_path = os.path.join(cls_path, cls)
        img_lst = os.listdir(data_path)
        for n, img in enumerate(img_lst):
            img_path = os.path.join(data_path, img)
            label_map[img_path] = label_to_index[cls]

    return label_map


def dict_to_tf_example(image_name,
                       dataset_directory,
                       label_map=None,
                       image_subdirectory='train'):
    """
    Args:
      image: a single image name
      dataset_directory: Path to root directory holding PCam dataset
      label_map: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        PCam dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by image is not a valid PNG
    """
    # full_path = os.path.join(dataset_directory, image_subdirectory, image_name)
    full_path = os.path.join(dataset_directory, image_name)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded = fid.read()
    encoded_io = io.BytesIO(encoded)
    image = Image.open(encoded_io)
    width, height = image.size
    format = image.format
    image_stat = ImageStat.Stat(image)
    mean = image_stat.mean
    std = image_stat.stddev
    key = hashlib.sha256(encoded).hexdigest()
    # if image_subdirectory.lower() == 'test':
    #     label = -1
    # else:
    #     label = int(label_map[image_name])
    label = int(label_map[full_path])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_utils.int64_feature(height),
        'image/width': dataset_utils.int64_feature(width),
        'image/filename': dataset_utils.bytes_feature(image_name.encode('utf8')),
        'image/fullpath': dataset_utils.bytes_feature(full_path.encode('utf8')),
        'image/source_id': dataset_utils.bytes_feature(image_name.encode('utf8')),
        'image/key/sha256': dataset_utils.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_utils.bytes_feature(encoded),
        'image/format': dataset_utils.bytes_feature(format.encode('utf8')),
        'image/class/label': dataset_utils.int64_feature(label),
        # 'image/text': dataset_util.bytes_feature('label_text'.encode('utf8'))
        'image/mean': dataset_utils.float_list_feature(mean),
        'image/std': dataset_utils.float_list_feature(std)
    }))
    return example


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter(FLAGS.output_path, options=options)

    # cls_lst = os.listdir(FLAGS.dataset_dir)
    dataset_lst = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_category)
    cls_lst = os.listdir(dataset_lst)
    cls_lst.sort()
    label_to_index = {}
    for i, cls in enumerate(cls_lst):
        cls_path = os.path.join(dataset_lst, cls)
        if os.path.isdir(cls_path):
            label_to_index[cls] = i

    label_map = get_label_map(label_to_index)

    random.shuffle(cls_lst)
    for i, cls in enumerate(cls_lst):
        cls_path = os.path.join(dataset_lst, cls)
        img_lst = os.listdir(cls_path)

        total = len(img_lst)
        for idx, image in enumerate(img_lst):
            if idx % 100 == 0:
                tf.compat.v1.logging.info('On image %d of %d', idx, total)

            tf_example = dict_to_tf_example(image, cls_path, label_map, FLAGS.dataset_category)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
