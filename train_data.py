from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random


MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]


class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, batch_size, num_classes, num_epochs, data_size, height, width):
        self.resize_h = height
        self.resize_w = width

        # self.target_dist = []
        # dist = 1.0 / num_classes
        # for i in range(num_classes):
        #     self.target_dist.append(round(dist, 2))

        self.dataset = tf.data.TFRecordDataset(tfrecord_path,
                                          compression_type='GZIP',
                                          num_parallel_reads=batch_size * 4)
        # dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        # The map transformation takes a function and applies it to every element
        # of the dataset.
        self.dataset = self.dataset.map(self.decode, num_parallel_calls=8)
        self.dataset = self.dataset.map(self.distort_image, num_parallel_calls=8)
        self.dataset = self.dataset.map(self.normalize, num_parallel_calls=8)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        self.dataset = \
            self.dataset.shuffle(buffer_size=(int(data_size * 0.4) + 3 * batch_size), seed=88)
        # buffer_size=(int(len(data_list) * 0.4) + 3 * batch_size)
        self.dataset = self.dataset.repeat()

        # resampling = \
        #     tf.data.experimental.rejection_resample(class_func=self.class_mapping_function,
        #                                             target_dist=self.target_dist)
        # self.dataset = self.dataset.apply(resampling)
        # # Return to the same Dataset shape as was the original input
        # self.dataset = self.dataset.map(lambda _, data: (data))

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


    def distort_image(self, filename, image, label):
        """Prepare one image for training.
        """
        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method based on the random number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = random.randint(0, 3)
        image = tf.image.resize(image, [self.resize_h, self.resize_w],
                                method=resize_method)
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly distort the colors.
        color_ordering = random.randint(0, 1)
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.2, upper=1.2)
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.2)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.2, upper=1.2)
            image = tf.image.random_hue(image, max_delta=0.1)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)

        return filename, image, label


    def normalize(self, filename, image, label):
        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        # return filename, tf.div(tf.subtract(image, MEAN), STD), label
        return filename, image, label


    def class_mapping_function(self, filename, image, label):
        """
            returns a function to be used with dataset.map() to return class numeric ID
            The function is mapping a nested structure of tensors (having shapes and types defined by dataset.output_shapes
            and dataset.output_types) to a scalar tf.int32 tensor. Values should be in [0, num_classes).
            """
        # For simplicity, trying to return the label itself as I assume its numeric...

        return label

