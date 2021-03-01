import tensorflow as tf


def dcg_preprocess(ex, z_size, mode='uniform'):
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = image * 2 - 1.0
    if mode == 'uniform':
        input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        input_z = tf.random.normal(shape=(z_size,))

    return input_z, image
