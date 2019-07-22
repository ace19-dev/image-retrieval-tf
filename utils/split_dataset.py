# ========================================================================
# Resize, Pad Image to Square Shape and Keep Its Aspect Ratio With Python
# ========================================================================

import os
import argparse
import sys

from PIL import Image, ImageFile
import shutil


import tensorflow as tf

FLAGS = None


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    img_lst = os.listdir(FLAGS.original_dir)
    # img_lst.sort()
    # print(img_lst)

    for img in img_lst:
        dir_name = img.split('.')[0].split('_')[1]
        new_dir_path = os.path.join(FLAGS.original_dir,dir_name)
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)

        img_path = os.path.join(FLAGS.original_dir, img)
        shutil.copyfile(img_path, os.path.join(FLAGS.target_dir, dir_name, img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--original_dir',
        type=str,
        default='/home/ace19/dl_data/materials/train',
        help='Where is image to load.')
    parser.add_argument(
        '--target_dir',
        type=str,
        default='/home/ace19/dl_data/materials/train',
        help='Where is resized image to save.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
