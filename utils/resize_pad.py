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

    desired_size = FLAGS.desired_size

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

                old_size = im.size  # old_size[0] is in (width, height) format

                ratio = float(desired_size) / max(old_size)
                new_size = tuple([int(x * ratio) for x in old_size])

                # use thumbnail() or resize() method to resize the input image
                # thumbnail is a in-place operation
                im.thumbnail(new_size, Image.ANTIALIAS)
                # im = im.resize(new_size, Image.ANTIALIAS)

                # create a new image and paste the resized on it
                new_im = Image.new("RGB", (desired_size, desired_size))
                new_im.paste(im, ((desired_size - new_size[0]) // 2,
                                  (desired_size - new_size[1]) // 2))

                outfile = os.path.join(target_dir, img)
                new_im.save(outfile)


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
        default='/home/ace19/dl_data/v2-plant-seedlings-dataset_thumbnail/classes',
        help='Where is resized image to save.')
    parser.add_argument(
        '--desired_size',
        type=int,
        default=224,
        help='how do you want image resize height, width.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
