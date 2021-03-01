import tensorflow_datasets as tfds
import dcgtest as dcgan

if __name__ == '__main__':
    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    mnist = mnist_bldr.as_dataset(shuffle_files=False)

    #dcgan.create_generator_test()
    dcgan.create_discriminator_test()