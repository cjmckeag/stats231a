import numpy as np


class TwoLayerNet(object):
    """
    A class to build a two layer neural network (one input layer, one hidden layer).
    ...

    Attributes
    ----------
    X : np.ndarray
        input dataset matrix where each row is a training example
    y : np.ndarray
        Output dataset matrix where each row is a training example
    activation_function : str
        activation_function function to use for hidden layer; takes 'sigmoid' or 'relu'
    learning_rate : float
        learning rate of weights to use for gradient descent

    Methods
    -------
    hidden_layer(activation_function, x, derivative=False):
        Provides either the sigmoid function or ReLU function to use in hidden layer.
    train():
        Performs forward propagation and gradient descent to learn weights and build the net.
    """

    def __init__(self, X, y, activation_function="sigmoid", learning_rate=1.0):
        """
        Constructs all the necessary attributes for the TwoLayerNet object.

        Parameters
        ----------
            X : np.ndarray
                input dataset matrix where each row is a training example
            y : np.ndarray
                Output dataset matrix where each row is a training example
            activation_function : str
                activation_function function to use for hidden layer, default sigmoid
            learning_rate : float
                learning rate of weights to use for gradient descent, default 1
        """
        self.X = X
        self.y = y
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    @staticmethod
    def hidden_layer(activation_function, x, derivative=False):
        """
        Provides either the sigmoid function or ReLU function to use in hidden layer.

        Parameters
        ----------
        activation_function : str
            specifies function
        x : np.ndarray
            inner product vector to apply function on
        derivative : bool
            whether to use derivative of function, default False

        Returns
        -------
        function output, dependent on parameters
        """
        if activation_function == "sigmoid":
            if derivative:
                return x * (1 - x)
            else:
                return 1 / (1 + np.exp(-x))
        elif activation_function == "relu":
            if derivative:
                return np.where(x > 0, 1, 0)
            else:
                return np.where(x > 0, x, 0)

    def train(self):
        """
        Performs backpropagation and gradient descent to learn weights and build the network

        Returns
        -------
        mae : float
            mean absolute error of final predictions
        """
        # seed random numbers to make calculation deterministic
        np.random.seed(1)
        # initialize weights randomly with mean 0
        p = self.X.shape[1]
        beta = 2 * np.random.random((p, 1)) - 1
        #beta = np.repeat(0.0,p).reshape(p,1)

        for i in range(10000):
            # forward propagation
            layer_1 = self.X
            layer_2 = self.hidden_layer(activation_function=self.activation_function,
                                        x=np.dot(layer_1, beta),
                                        derivative=False)

            # how much did we miss?
            layer_2_error = self.y - layer_2

            # multiply how much we missed by the
            # slope of the activation function at the values in layer_2
            layer_2_delta = layer_2_error * self.hidden_layer(activation_function=self.activation_function,
                                                              x=layer_2,
                                                              derivative=True)

            # update weights
            beta += self.learning_rate * np.dot(layer_1.T, layer_2_delta)

        mae = np.mean(np.abs(layer_2_error))
        print("Error:" + str(mae))
        return mae