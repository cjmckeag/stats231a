import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import the fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0


class Classification:
    """
    Trains a neural network model to classify images.

    Methods
    -------
    plot_image(i, predictions_array, true_label, img):
        Utility method for plotting image.
    plot_value_array(i, predictions_array, true_label):
        Utility method to plot array of values.
    train(self):
        Builds, compiles, trains, and evaluates a Sequential neural network.
    predict(self, model):
        Makes predictions on test images and plots an array of predicted labels and the true images.
    """

    def __init__(self, train_images, train_labels, test_images, test_labels, activation='relu', num_nodes1=128,
                 add_layer=False):
        """
        Constructs all the necessary attributes for the Classification object.
        :param train_images: images to train on
        :param train_labels: training labels
        :param test_images: images to test on
        :param test_labels: testing labels
        :param activation: activation function to use (e.g., relu, sigmoid, linear, softmax...)
        :param num_nodes1: number of nodes in first dense layer
        :param add_layer: whether or not to add another dense layer
        """
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.activation = activation
        self.num_nodes1 = num_nodes1
        self.add_layer = add_layer

    def __call__(self):
        """

        :return:
        """
        model = self.train()
        self.predict(model)

    @staticmethod
    def plot_image(i, predictions_array, true_label, img):
        """
        Utility method for plotting image.
        :param i: index of image
        :param predictions_array: array of predictions for image
        :param true_label: true labels of all images
        :param img: all test images
        :return: nothing
        """
        # select i'th image from all test images and all true labels
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            # BLUE IS RIGHT
            color = 'blue'
        else:
            # RED IS WRONG
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    @staticmethod
    def plot_value_array(i, predictions_array, true_label):
        """
        Utility method to plot array of values.
        :param i: index of image
        :param predictions_array: array of predictions for image
        :param true_label: true image label
        :return: nothing
        """
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def train(self):
        """
        Builds, compiles, trains, and evaluates a Sequential neural network.
        :return: model that is trained and evaluated
        """
        # build and compile the model
        # chaining together simple layers
        # if want to add another dense layer
        if self.add_layer:
            model = tf.keras.Sequential([
                # transforms the format of the images from a two-dimensional array to a one-dimensional array
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                # three fully connected layers
                tf.keras.layers.Dense(self.num_nodes1, activation=self.activation),
                tf.keras.layers.Dense(64, activation=self.activation),
                tf.keras.layers.Dense(10)
            ])
        # if not adding another dense layer
        else:
            model = tf.keras.Sequential([
                # transforms the format of the images from a two-dimensional array to a one-dimensional array
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                # two fully connected layers
                tf.keras.layers.Dense(self.num_nodes1, activation=self.activation),
                tf.keras.layers.Dense(10)
            ])

        # compile step
        # optimizer: how the model is updated based on the data it sees and its loss function
        # loss: measures how accurate the model is during training
        # metrics: used to monitor the training and testing steps
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # train the model
        # feeds the training data, model learns to associate images and labels,
        # ask the model to make predictions about a test set, verify that the predictions match the labels
        model.fit(self.train_images, self.train_labels, epochs=10)

        # evaluate
        # compare how the model performs on the test dataset
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

        return model

    def predict(self, model):
        """
        Makes predictions on test images and plots an array of predicted labels and the true images.
        :param model: model trained with train() method
        :return: nothing
        """
        # make predictions
        # attach a softmax layer to convert the logits to probabilities, which are easier to interpret
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        predictions = probability_model.predict(self.test_images)

        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions[i], self.test_labels, self.test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions[i], self.test_labels)
        plt.tight_layout()
        plt.show()
