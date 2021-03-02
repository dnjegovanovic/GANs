import models.dcgan as dcgan
import utils.dcgandataprep as dprep
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time

if tf.test.is_gpu_available():
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'

print(device_name)


def create_generator_test():
    gen_model = dcgan.make_dcgan_generator()
    gen_model.summary()


def create_discriminator_test():
    disc_model = dcgan.make_dcgan_discriminator()
    disc_model.summary()


def train_dcgan():
    """
    Implementaion of WGAN-GP model for training DCGAN
    """

    np.random.seed(1)
    tf.random.set_seed(1)

    num_epoch = 100
    batch_size = 128
    image_size = (28, 28)
    mode_z = 'uniform'
    l_gp = 10.0
    z_size = 20

    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    mnist = mnist_bldr.as_dataset(shuffle_files=False)

    mnist_trainset = mnist['train']
    mnist_trainset = mnist_trainset.map(lambda ex: dprep.dcg_preprocess(ex, mode=mode_z))

    # Test passing data
    mnist_trainset = mnist_trainset.shuffle(10000)
    mnist_trainset = mnist_trainset.batch(batch_size, drop_remainder=True)

    with tf.device(device_name):
        gen_model = dcgan.make_dcgan_generator()
        gen_model.build(input_shape=(None, z_size))

        disc_model = dcgan.make_dcgan_discriminator()
        disc_model.build(input_shape=(None, np.prod(image_size)))

        # optimizers
        g_opt = tf.keras.optimizers.Adam(0.0002)
        d_opt = tf.keras.optimizers.Adam(0.0002)

        if mode_z == 'uniform':
            fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
        elif mode_z == 'normal':
            fixed_z = tf.random.normal(shape=(batch_size, z_size))

        all_losses = []
        epoch_samples = []
        start_time = time.time()

        for epoch in range(1, num_epoch + 1):
            epoch_losses = []

            for i, (input_z, input_real) in enumerate(mnist_trainset):
                with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                    g_output = gen_model(input_z, training=True)
                    d_critics_real = disc_model(input_real, training=True)
                    d_critics_fake = disc_model(g_output, training=True)

                    # Compute generator loss
                    g_loss = tf.math.reduce_mean(d_critics_fake)

                    # Compute disc loss
                    d_loss_real = tf.math.reduce_mean(d_critics_real)
                    d_loss_fake = tf.math.reduce_mean(d_critics_fake)
                    d_loss = d_loss_real + d_loss_fake

                    # Gradient penalty
                    with tf.GradientTape() as gp_tape:
                        alpha = tf.random.uniform(shape=(d_critics_real.shape[0], 1, 1, 1),
                                                  minval=0.0, maxval=1.0)
                        interpolated = (alpha * input_real + (1 - alpha) * g_output)
                        gp_tape.watch(interpolated)
                        d_critics_intp = disc_model(interpolated)

                    grads_intp = gp_tape.gradient(d_critics_intp, [interpolated, ])[0]
                    grads_intp_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
                    grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))

                    d_loss = d_loss + l_gp * grad_penalty

                # Optimization: compute grad and applay them
                d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
                d_opt.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables))

                g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
                g_opt.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))

                epoch_losses.append((g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy()))

            all_losses.append(epoch_losses)

            print('Epoch {:-3d} | ET {:.2f} min | Avg Losses >>'
                  ' G/D {:6.2f}/{:6.2f} [D-Real: {:6.2f} D-Fake: {:6.2f}]'
                  .format(epoch, (time.time() - start_time) / 60,
                          *list(np.mean(all_losses[-1], axis=0)))
                  )

            epoch_samples.append(dprep.create_samples(gen_model, fixed_z, batch_size,image_size).numpy())
        dprep.visualize(all_losses,epoch_samples)
