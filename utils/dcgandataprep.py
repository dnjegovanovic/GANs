import tensorflow as tf
import matplotlib as plt
import itertools
import numpy as np

def dcg_preprocess(ex, z_size=20, mode='uniform'):
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = image * 2 - 1.0
    if mode == 'uniform':
        input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        input_z = tf.random.normal(shape=(z_size,))

    return input_z, image


def create_samples(g_model, input_z, batch_size, image_size):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0

def visualize(all_losses, epoch_samples):
    fig = plt.figure(figsize=(8, 6))

    ## Plotting the losses
    ax = fig.add_subplot(1, 1, 1)
    g_losses = [item[0] for item in itertools.chain(*all_losses)]
    d_losses = [item[1] for item in itertools.chain(*all_losses)]
    plt.plot(g_losses, label='Generator loss', alpha=0.95)
    plt.plot(d_losses, label='Discriminator loss', alpha=0.95)
    plt.legend(fontsize=20)
    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Loss', size=15)

    epochs = np.arange(1, 101)
    epoch2iter = lambda e: e * len(all_losses[-1])
    epoch_ticks = [1, 20, 40, 60, 80, 100]
    newpos = [epoch2iter(e) for e in epoch_ticks]
    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(epoch_ticks)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.set_xlabel('Epoch', size=15)
    ax2.set_xlim(ax.get_xlim())
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    # plt.savefig('images/ch17-wdcgan-learning-curve.pdf')
    plt.show()

    selected_epochs = [1, 2, 4, 10, 50, 100]
    fig = plt.figure(figsize=(10, 14))
    for i, e in enumerate(selected_epochs):
        for j in range(5):
            ax = fig.add_subplot(6, 5, i * 5 + j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.text(
                    -0.06, 0.5, 'Epoch {}'.format(e),
                    rotation=90, size=18, color='red',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax.transAxes)

            image = epoch_samples[e - 1][j]
            ax.imshow(image, cmap='gray_r')

    plt.savefig('images/ch17-wdcgan-samples.png')
    plt.show()