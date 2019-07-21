# ========================================================================
# Resize, Pad Image to Square Shape and Keep Its Aspect Ratio With Python
# ========================================================================

import os
import argparse
import sys

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf

FLAGS = None


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

    cls_lst = os.listdir(FLAGS.original_dir)
    cls_lst.sort()
    # print(cls_lst)

    size = FLAGS.desired_size.split(',')
    size = tuple(int(s) for s in size)

    for classname in cls_lst:
        class_path = os.path.join(FLAGS.original_dir, str(classname))
        data_category = os.listdir(class_path)
        # total = len(data_cate)
        for cate in data_category:
            target_dir = os.path.join(FLAGS.target_dir, str(classname), cate)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            img_dir = os.path.join(class_path, cate)
            images = os.listdir(img_dir)
            total = len(images)
            for idx, img in enumerate(images):
                if idx % 100 == 0:
                    tf.logging.info('On image %d of %d', idx, total)

                image_path = os.path.join(img_dir, img)
                im = Image.open(image_path)
                im.thumbnail(size, Image.ANTIALIAS)
                # im.resize(size)
                outfile = os.path.join(target_dir, img)
                im.save(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--original_dir',
        type=str,
        default='/home/ace19/dl_data/v2-plant-seedlings-dataset/classes',
        help='Where is image to load.')
    parser.add_argument(
        '--target_dir',
        type=str,
        default='/home/ace19/dl_data/v2-plant-seedlings-dataset_resized/classes',
        help='Where is resized image to save.')
    parser.add_argument(
        '--desired_size',
        type=str,
        default='224,224',
        help='how do you want image resize height, width.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
