import models.dcgan as dcgan


def create_generator_test():
    gen_model = dcgan.make_dcgan_generator()
    gen_model.summary()

def create_discriminator_test():
    disc_model = dcgan.make_dcgan_discriminator()
    disc_model.summary()