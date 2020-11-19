import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras import layers
import time

(train_mnist_images, train_mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_mnist_images = train_mnist_images.reshape(train_mnist_images.shape[0], 28, 28, 1).astype('float32')
train_mnist_images = (train_mnist_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_mnist_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


class GAN:
    """
    what it does

    Methods
    -------
    make_generator_model(self):
        Builds the generator model, which learns to create "fake" images that look real. The generator will generate
         handwritten digits resembling the MNIST data.
    make_discriminator_model(self):
        Builds the discriminator model, which learns to tell real images apart from fakes. It is a CNN-based image
         classifier. The model will be trained to output positive values for real images, and negative values for fake
          images.
    discriminator_loss(self, real_output, fake_output):
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares
         the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake
          (generated) images to an array of 0s.
    generator_loss(self, fake_output):
        The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator
         is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the
          discriminators decisions on the generated images to an array of 1s.
    train_step(self, images, generator, discriminator, generator_optimizer, discriminator_optimizer):
        Performs one step of training. Generator produces an image, discriminator classifies between reals and fakes,
         loss is computed for each model, then gradients are used to update each model.
    train(self, dataset, generator, discriminator, generator_optimizer, discriminator_optimizer):
        Uses train_step() method to iterate over dataset and learn generator+discriminator. Generates image after the
         final epoch.
    generate_and_save_images(model, epoch, test_input):
        Utility method to plot generator images.
    display_image(epoch_no):
        Display a single image using the epoch number.
    """

    def __init__(self, train_dataset, add_layers=False, epochs=50, noise_dim=100, num_examples_to_generate=16,
                 batch_size=256):
        """
        Constructs all necessary parameters for GAN object.
        :param train_dataset: properly shaped training dataset
        :param add_layers: whether or not to add a chunk of layers to both generator and discriminator models
        :param epochs: number of epochs for training
        :param noise_dim: dimension of noise variable
        :param num_examples_to_generate: number of examples to generate (for gif)
        :param batch_size: size of training batches
        """
        self.train_dataset = train_dataset
        self.add_layers = add_layers
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate
        self.batch_size = batch_size
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

    def __call__(self):
        """
        When called, the models are built and trained, then a resulting example is displayed.
        :return: nothing
        """
        generator = self.make_generator_model()
        discriminator = self.make_discriminator_model()
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer)
        self.display_image(self.epochs)

    def make_generator_model(self):
        """
        Builds the generator model, which learns to create "fake" images that look real. The generator will generate
         handwritten digits resembling the MNIST data.
        :return: the generator model
        """
        model = tf.keras.Sequential()
        # dense layer takes random seed as input
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        # upsampling layer to produce an image from a seed
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # upsample several times until reach the desired image size of 28x28x1
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.add_layers:
            model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 14, 14, 32)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        # output layer activation is tanh
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        """
        Builds the discriminator model, which learns to tell real images apart from fakes. It is a CNN-based image
         classifier. The model will be trained to output positive values for real images, and negative values for fake
          images.
        :return: the discriminator model
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        if self.add_layers:
            model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares
         the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake
          (generated) images to an array of 0s.
        :param real_output: discriminator's predictions on real images
        :param fake_output: discriminator's predictions on fake (generated) images
        :return: sum cross-entropy of real predictions and fake predictions
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        """
        The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator
         is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the
          discriminators decisions on the generated images to an array of 1s.
        :param fake_output: discriminator's predictions on (generated) fake images
        :return: cross-entropy of fake predictions and array of 1s
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images, generator, discriminator, generator_optimizer, discriminator_optimizer):
        """
        Performs one step of training. Generator produces an image, discriminator classifies between reals and fakes,
         loss is computed for each model, then gradients are used to update each model.
        :param images: images for one batch
        :param generator: current step generator
        :param discriminator: current step discriminator
        :param generator_optimizer: optimizer that updates the generator
        :param discriminator_optimizer: optimizer that updates the optimizer
        :return: nothing
        """
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # begins with generator receiving a random seed as input, produces an image
            generated_images = generator(noise, training=True)

            # discriminator is used to classify real images and fake images
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            # loss is calculated for each of these models
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # gradients are used to update the generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, generator, discriminator, generator_optimizer, discriminator_optimizer):
        """
        Uses train_step() method to iterate over dataset and learn generator+discriminator. Generates image after the
         final epoch.
        :param dataset: training set
        :param generator: initialized generator
        :param discriminator: initialized discriminator
        :param generator_optimizer: optimizer that updates the generator
        :param discriminator_optimizer: optimizer that updates the discriminator
        :return: nothing
        """
        for epoch in range(self.epochs):
            print('Training epoch {}'.format(epoch + 1))
            start = time.time()
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch, generator, discriminator, generator_optimizer,
                                                      discriminator_optimizer)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            # self.generate_and_save_images(generator,
            #                              epoch + 1,
            #                              self.seed)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        # self.generate_and_save_images(generator,
        #                              self.epochs,
        #                              self.seed)
        print('Final Generator Loss: {}'.format(gen_loss))
        print('Final Discriminator Loss: {}'.format(disc_loss))

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        """
        Utility method to plot generator images.
        :param model: generator model
        :param epoch: epoch num of image to plot
        :param test_input: input seed
        :return: nothing
        """
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    @staticmethod
    def display_image(epoch_no):
        """
        Display a single image using the epoch number.
        :param epoch_no: epoch number
        :return: displayed image
        """
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
