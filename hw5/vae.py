from abc import ABC

from IPython import display

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time


# define the encoder and decoder networks
class CVAE(tf.keras.Model, ABC):
    """
    Builds the Convolutional Variational Autoencoder. Uses tf.keras.Sequential to define the encoder and decoder
     networks. Each network is a small ConvNet.
    """
    def __init__(self, latent_dim):
        """
        Constructs necessary parameters for CVAE.
        :param latent_dim: dimension of latent sample
        """
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        # encoder takes an observation as input and outputs a set of parameters for specifying the conditional
        # distribution of the latent representation
        # architecture: two convolutional layers followed by a fully-connected layer
        # defines approx posterior distr p(z|x)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )
        # decoder takes a latent sample as input and outputs the parameters for a conditional distribution of
        # the observation
        # architecture: fully-connected layer followed by three convolutional transpose layers
        # defines conditional distr p(x|z)
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        """
        Samples standard normal RVs and passes them through decoder
        :param eps: optional RV input
        :return: decoded sample
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        """
        Passes input through encoder, splits into sub-tensors
        :param x: input to encode
        :return: mean, logvar
        """
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        """
        Generates normal RVs then scales them with inputted variance and shifts them with inputted mean.
        Approximates z using the decoder parameters and an epsilon.
        :param mean: shift
        :param logvar: scale
        :return: reparameterized vector
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """
        Passes input through decoder.
        :param z: input
        :param apply_sigmoid: bool, whether to apply sigmoid activation
        :return: logit probs
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class VAE:
    """
    Trains a Variational Autoencoder on the MNIST dataset. The VAE maps the input data into the parameters of a
     probability distribution, such as the mean and variance of a Gaussian. This approach produces a continuous,
      structured latent space, which is useful for image generation.
    """
    def __init__(self, epochs=10, latent_dim=2, num_examples_to_generate=16, batch_size=32):
        """
        Constructs necessary parameters for VAE.
        :param epochs: num epochs to train
        :param latent_dim: dimension of latent sample
        :param num_examples_to_generate: num examples to generate
        :param batch_size: size of training batches
        """
        # define the optimizer
        # optimize the single sample MC estimate of the ELBO
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # training
        self.epochs = epochs
        # set the dimensionality of the latent space to a plane for visualization later
        self.latent_dim = latent_dim
        self.num_examples_to_generate = num_examples_to_generate
        self.batch_size = batch_size

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        self.random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, self.latent_dim])
        self.model = CVAE(self.latent_dim)

    def __call__(self):
        """
        Generates training and testing data, trains the network, displays a generated image from the last training
         epoch, then plots images decoded from the latent space.
        :return: nothing
        """
        self.train_dataset, self.test_dataset = self.get_data()
        self.train()
        # display a generated image from the last training epoch
        plt.imshow(self.display_image(self.epochs))
        plt.axis('off')

        # plots n x n digit images decoded from the latent space
        self.plot_latent_images(20)

    @staticmethod
    def preprocess_images(images):
        """
        Represent each pixel with a Bernoulli distribution and statically binarize the data.
        :param images: array of size 784 vectors
        :return: binarized image data
        """
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')

    def get_data(self):
        """
        Loads the MNIST dataset. Each image is originally a vector of 784 integers, each between 0 and 255,
         representing the intensity of a pixel.
        :return: preprocessed train_dataset, test_dataset
        """
        # load the MNIST dataset
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

        train_images = self.preprocess_images(train_images)
        test_images = self.preprocess_images(test_images)

        train_size = 60000
        batch_size = self.batch_size
        test_size = 10000

        # use tf.data to batch and shuffle data
        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                         .shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(test_size).batch(batch_size))

        return train_dataset, test_dataset

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        """
        Loss function
        :param sample: inputted sample
        :param mean: log normal mean
        :param logvar: lognorm variance to scale data
        :param raxis: axis to sum on
        :return: sum of elements of log normal pdf
        """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        """
        Computes ELBO loss.
        :param x: input
        :return: Monte Carlo estimate of ELBO
        """
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, x):
        """
        Executes one training step and returns the loss. This function computes the loss and gradients,
        and uses the latter to update the model's parameters.
        :param x: training input
        :return: nothing
        """
        # compute loss
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        # compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # update model parameters
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def generate_and_save_images(self, epoch, test_sample):
        """
        Samples a set of latent vectors from unit Gaussian prior which is then passed to the generator. The generator
         converts the latent sample to logits of the observation. The probabilities of the Bernoulli are then plotted.
        :param epoch: training epoch num
        :param test_sample: test sample
        :return: nothing
        """
        mean, logvar = self.model.encode(test_sample)
        # z is shape mean.shape
        z = self.model.reparameterize(mean, logvar)
        predictions = self.model.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(self):
        """
        Performs all training steps, computing loss at each epoch.
        :return: nothing
        """
        # Pick a sample of the test set for generating output images
        assert self.batch_size >= self.num_examples_to_generate
        # iterate over the dataset
        for test_batch in self.test_dataset.take(1):
            test_sample = test_batch[0:self.num_examples_to_generate, :, :, :]
            self.generate_and_save_images(0, test_sample)

            for epoch in range(1, self.epochs + 1):
                print('Training Epoch {} ...'.format(epoch))
                # perform and time all training steps for dataset
                start_time = time.time()
                for train_x in self.train_dataset:
                    self.train_step(train_x)
                end_time = time.time()

                # compute loss on test
                loss = tf.keras.metrics.Mean()
                for test_x in self.test_dataset:
                    loss(self.compute_loss(test_x))
                elbo = -loss.result()
                display.clear_output(wait=False)
                print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                      .format(epoch, elbo, end_time - start_time))
                self.generate_and_save_images(epoch, test_sample)

    @staticmethod
    def display_image(epoch_no):
        """
        Displays a generated image from the last training epoch.
        :param epoch_no: epoch number
        :return: generated image
        """
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    def plot_latent_images(self, n, digit_size=28):
        """
        Displays a 2D manifold of digits from the latent space. Plots n x n digit images decoded from the latent space.
        :param n: size of images
        :param digit_size: (image width)/n
        :return: nothing
        """

        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = digit_size * n
        image_height = image_width
        image = np.zeros((image_height, image_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_decoded = self.model.sample(z)
                digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
                image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='Greys_r')
        plt.axis('Off')
        plt.show()