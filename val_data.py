from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random


MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

RANDOM_CROP_SIZE = 224


class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, batch_size, num_classes, num_epochs, data_size, height, width):
        self.resize_h = height
        self.resize_w = width

        self.dataset = tf.data.TFRecordDataset(tfrecord_path,
                                          compression_type='GZIP',
                                          num_parallel_reads=batch_size * 4)
        # self.dataset = self.dataset.map(self._parse_func, num_parallel_calls=8)
        # The map transformation takes a function and applies it to every element
        # of the self.dataset.
        self.dataset = self.dataset.map(self.decode, num_parallel_calls=8)
        self.dataset = self.dataset.map(self.eval, num_parallel_calls=8)
        # self.dataset = self.dataset.map(self.tencrop, num_parallel_calls=8)
        self.dataset = self.dataset.map(self.normalize, num_parallel_calls=8)

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        self.dataset = self.dataset.shuffle(buffer_size=(int(data_size * 0.4) + 3 * batch_size))
        # self.dataset = self.dataset.shuffle(1000 + 3 * batch_size)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(batch_size)


    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image/fullpath': tf.io.FixedLenFeature([], tf.string),
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            })

        filename = features['image/fullpath']
        # Convert from a scalar string tensor to a float32 tensor with shape
        image_decoded = tf.image.decode_png(features['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['image/class/label'], tf.int64)

        return filename, image, label


    def eval(self, filename, image, label):
        """Prepare one image for evaluation.
        """
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [self.resize_h, self.resize_w])
        image = tf.squeeze(image, [0])

        return filename, image, label


    def tencrop(self, filename, image, label):
        """Prepare one image for TenCrop
        """
        # Resize the image to the height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [self.resize_h, self.resize_w])
        image = tf.squeeze(image, [0])

        flip_mode = random.randint(0, 1)
        images = []
        for i in range(5):
            image = tf.random_crop(image, [RANDOM_CROP_SIZE, RANDOM_CROP_SIZE, 3])
            image = tf.cast(image, tf.float32)
            images.append(image)
            if flip_mode == 0:
                images.append(tf.image.random_flip_left_right(image))
            elif flip_mode == 1:
                images.append(tf.image.random_flip_up_down(image))

        return filename, tf.stack(images), label


    def normalize(self, filename, image, label):
        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        # return filename, tf.div(tf.subtract(image, MEAN), STD), label
        return filename, image, label
