import tensorflow as tf
import numpy as np
from PIL import Image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def save_image(filepath, input_images):
    input_images = np.squeeze(input_images)
    img = Image.fromarray(input_images.astype('uint8')).convert('L')
    img.save(filepath, 'png')

def flip_image(image):
    return tf.image.flip_left_right(image)


def rescale(image, label, height, width, scale):
    image = tf.to_float(image)
    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)
    new_height = tf.to_int32(tf.to_float(height) * scale)
    new_width = tf.to_int32(tf.to_float(width) * scale)
    new_image = tf.image.resize_bilinear(image, [new_height, new_width])
    new_label = tf.image.resize_nearest_neighbor(label, [new_height, new_width])

    return new_image, new_label


def random_crop_and_pad(image, label, crop_height, crop_width, ignore_label=0):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label
    image_label = tf.concat([image, label], axis=3)
    image_shape = tf.shape(image)
    image_label_pad = tf.image.pad_to_bounding_box(image_label, 0, 0, tf.maximum(crop_height, image_shape[1]),
                                                   tf.maximum(crop_width, image_shape[2]))
    image_channels = tf.shape(image)[-1]
    image_label_pad = tf.squeeze(image_label_pad, axis=0)
    image_label_crop = tf.random_crop(image_label_pad, [crop_height, crop_width, 4])
    image_crop = image_label_crop[:, :, :image_channels]
    label_crop = image_label_crop[:, :, image_channels:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    image_crop.set_shape((crop_height, crop_width, 3))
    label_crop.set_shape((crop_height, crop_width, 1))

    return image_crop, label_crop


def preprocess_for_train(image, label):
    image_shape = tf.shape(image)
    image_height, image_width = image_shape[0], image_shape[1]

    # random flipping
    coin = tf.to_float(tf.random_uniform([1]))[0]
    image, label = tf.cond(tf.greater_equal(coin, 0.5),
                           lambda: (flip_image(image), flip_image(label)),
                           lambda: (image, label))

    scale = tf.random_uniform(shape=[1], minval=0.5, maxval=2)[0]
    image, label = rescale(image, label, image_height, image_width, scale)

    image, label = random_crop_and_pad(image, label, 224, 224)

    # rgb to gbr
    image = tf.reverse(image, axis=[-1])
    image -= IMG_MEAN

    return image, label


def preprocess_for_test(image, label):
    image = tf.to_float(image)
    image = tf.reverse(image, axis=[-1])
    image -= IMG_MEAN
    image = tf.expand_dims(image, axis=0)
    label = tf.expand_dims(label, axis=0)

    return image, label


def preprocess_image(image, label, is_training=False):
    if is_training:
        return preprocess_for_train(image, label)
    else:
        return preprocess_for_test(image, label)
